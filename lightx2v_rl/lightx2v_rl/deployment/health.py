"""Direct HTTP checks for internal reward services, independent of web proxies."""

import argparse
import concurrent.futures
import time
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener


def direct_urlopen(request, timeout):
    """Open an internal service request without changing process proxy settings."""
    return build_opener(ProxyHandler({})).open(request, timeout=timeout)


def health_failures(urls, timeout=3.0):
    """Check all endpoints concurrently and return each failure with its URL."""
    urls = tuple(urls)
    if not urls:
        raise ValueError("At least one reward health endpoint is required.")

    def check(url):
        try:
            with direct_urlopen(url, timeout=timeout) as response:
                if response.status != 200:
                    return f"{url}: HTTP {response.status}"
        except HTTPError as exc:
            try:
                return f"{url}: HTTP {exc.code} ({exc.reason})"
            finally:
                exc.close()
        except (OSError, HTTPException, ValueError) as exc:
            return f"{url}: {type(exc).__name__}: {exc}"
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as pool:
        return [failure for failure in pool.map(check, urls) if failure is not None]


def _report(failures):
    for failure in failures:
        print(f"Reward readiness failed: {failure}", flush=True)


class HTTPReadiness:
    """Callable readiness check with periodic diagnostics while services start."""

    def __init__(self, urls, timeout=3.0, report_interval=30.0):
        self.urls = tuple(urls)
        self.timeout = timeout
        self.report_interval = report_interval
        self._last_report = None

    def __call__(self):
        failures = health_failures(self.urls, timeout=self.timeout)
        if not failures:
            self._last_report = None
            return True
        now = time.monotonic()
        if self._last_report is None or now - self._last_report >= self.report_interval:
            _report(failures)
            self._last_report = now
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Print failed endpoints and their HTTP status or exception.")
    parser.add_argument("urls", nargs="+", help="Internal reward health URLs")
    args = parser.parse_args(argv)
    failures = health_failures(args.urls)
    if args.report:
        _report(failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
