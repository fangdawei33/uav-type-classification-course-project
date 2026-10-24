"""Rate-limited HTTP with robots checks and bounded retries."""
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser
import requests
from .common import OUT, append_log, save_json, now

class Web:
    def __init__(self, cfg, allowed_hosts=None):
        # Each collector supplies an explicit host allowlist; legacy Commons callers
        # retain their original two-host scope. Redirects use the same allowlist.
        self.allowed_hosts = frozenset(allowed_hosts if allowed_hosts is not None else
            {"commons.wikimedia.org", "upload.wikimedia.org"})
        if not self.allowed_hosts:
            raise ValueError("Crawler host allowlist must not be empty")
        self.cfg = dict(cfg); self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg["user_agent"]
        self.robots = {}; self.last = 0.0

    def raw(self, url, max_bytes=25_000_000):
        # Check each redirect destination before fetching it. Bounded retries only.
        for hop in range(5):
            for attempt in range(3):
                time.sleep(max(0, self.cfg["request_delay_seconds"] - (time.monotonic() - self.last)))
                self.last = time.monotonic()
                try:
                    response = self.session.get(url, timeout=(15, 60), stream=True, allow_redirects=False)
                    append_log("requests", {"url": url, "status": response.status_code, "attempt": attempt})
                    if response.status_code in [429, 500, 502, 503, 504]:
                        response.close()
                        if attempt == 2: raise RuntimeError("网站限流或服务不可用；请稍后重跑，已采集结果保留。")
                        time.sleep(5 * (attempt + 1)); continue
                    break
                except requests.RequestException:
                    if attempt == 2: raise
                    time.sleep(3 * (attempt + 1))
            if response.status_code in [301, 302, 303, 307, 308]:
                from urllib.parse import urljoin
                target = urljoin(url, response.headers["Location"]); response.close()
                self.allowed(target); url = target; continue
            response.raise_for_status()
            try:
                chunks = []; total = 0
                for chunk in response.iter_content(65536):
                    total += len(chunk)
                    if total > max_bytes: raise ValueError("response too large")
                    chunks.append(chunk)
                return b"".join(chunks)
            finally: response.close()
        raise RuntimeError("Too many redirects")

    def allowed(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts:
            raise ValueError("Unapproved crawler host: " + url)
        origin = parsed.scheme + "://" + parsed.netloc
        if origin not in self.robots:
            robots_url = origin + "/robots.txt"
            try:
                content = self.raw(robots_url, 2_000_000).decode("utf-8", "replace")
                parser = RobotFileParser(); parser.parse(content.splitlines())
            except requests.HTTPError as exc:
                # RFC 9309 section 2.3.1.3 permits access when robots.txt is
                # unavailable. Accept only an explicit 404 here; do not treat
                # 403, rate limiting, network failure or 5xx as permission.
                if exc.response is None or exc.response.status_code != 404:
                    raise RuntimeError("无法确认 robots 规则，停止此来源：" + origin) from exc
                exc.response.close()
                content = "# robots.txt returned HTTP 404; no exclusion file was published.\nUser-agent: *\nDisallow:"
                parser = RobotFileParser()
                parser.parse(content.splitlines())
                receipt = {"observed_at": now(), "url": robots_url, "http_status": 404,
                           "decision": "allow_missing_robots_404_per_RFC9309_2.3.1.3",
                           "generated_parser_policy_not_a_downloaded_robots_file": True,
                           "reference": "https://www.rfc-editor.org/rfc/rfc9309.html#section-2.3.1.3"}
                save_json(OUT / "logs" / (parsed.hostname + "_robots_status.json"), receipt)
                append_log("robots", receipt)
            except Exception as exc:
                raise RuntimeError("无法确认 robots 规则，停止此来源：" + origin) from exc
            self.robots[origin] = parser
            (OUT / "logs" / (parsed.hostname + "_robots.txt")).write_text(content, encoding="utf-8")
        parser = self.robots[origin]
        if not parser.can_fetch(self.cfg["user_agent"], url):
            raise PermissionError("robots 不允许访问：" + url)
        delay = parser.crawl_delay(self.cfg["user_agent"]) or parser.crawl_delay("*")
        if delay: self.cfg["request_delay_seconds"] = max(self.cfg["request_delay_seconds"], delay)

    def get(self, url, max_bytes=25_000_000):
        self.allowed(url)
        return self.raw(url, max_bytes)
