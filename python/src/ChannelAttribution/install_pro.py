def __request_token_channelattributionpro(
        email: str,
        endpoint: str = "https://app.channelattribution.io/genpkg/generate_token.php",
        timeout: int = 60,
        verify_ssl: bool = True
    ) -> str:
    
        '''
        Send an email address to ChannelAttribution Pro's `generate_token.php` endpoint and
        return the server's response body.
        
        Parameters
        ----------
        email : str
            Target email address to which the token should be sent. Must be **non-empty**.
            (No syntactic validation is performed by this function.)
        endpoint : str, default "https://app.channelattribution.io/genpkg/generate_token.php"
            Full URL of the token-generation PHP endpoint. You can override this for testing.
        timeout : int, default 60
            Timeout in seconds applied to the HTTP request.
        verify_ssl : bool, default True
            Whether to verify the server's TLS certificate. Set to ``False`` only in
            controlled testing environments.
        
        Returns
        -------
        str
            The exact response body (trimmed) returned by the server. Typical values include:
            - "We’ve sent the token to your email address. ..."
            - "Token already generated"
            - "Provider not admitted"
            - "mail not valid"
            - "db query error", "db connection error", etc.
        
        Raises
        ------
        ValueError
            If ``email`` is empty.
        RuntimeError
            For network/TLS issues (connection errors, DNS failure, timeouts, SSL problems),
            with a message prefixed by ``"network_or_ssl_error:"`` or ``"request_error:"``.
        
        Behavior
        --------
        - **Prefers POST** (form data). If the server rejects the method (e.g., HTTP 405/403
          with a “method” hint in the body), it **retries with GET**.
        - Does **not** raise for non-2xx HTTP statuses; it returns the body as-is so the caller
          can show the server’s message to the user.
        - If the ``requests`` package is missing, it is installed into the **current interpreter**
          (respecting Debian/Ubuntu PEP 668 by setting ``PIP_BREAK_SYSTEM_PACKAGES=1`` only when
          not in a virtualenv).
        
        Notes
        -----
        - To validate the email syntax before calling this function, do it in the caller
          (e.g., `from email.utils import parseaddr` and check that `parseaddr(email)[1]` is non-empty).
        - Honors system proxy settings if they are configured for Python/OS.
        
        Examples
        --------
        Basic usage
        
        >>> from ChannelAttribution import request_token_channelattributionpro
        >>> request_token_channelattributionpro("john.black@company.com")
        '''
    
        import sys, subprocess
        try:
            import requests
        except ImportError:
            import os
            env = dict(os.environ)
            if sys.prefix == sys.base_prefix:  # not in venv → allow system install (PEP 668)
                env.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"], env=env)
        
        import requests
        from requests.exceptions import RequestException, Timeout, SSLError
    
        if not email:
            raise ValueError("email must be a non-empty string")
    
        try:
            # Prefer POST
            resp = requests.post(
                endpoint,
                data={"email": email},
                timeout=timeout,
                verify=verify_ssl,
                allow_redirects=True,
                headers={"User-Agent": "capro-token-client/1.0"}
            )
    
            # If server disallows POST (rare), retry with GET
            if resp.status_code in (405, 403) and "method" in (resp.text or "").lower():
                resp = requests.get(
                    endpoint,
                    params={"email": email},
                    timeout=timeout,
                    verify=verify_ssl,
                    allow_redirects=True,
                    headers={"User-Agent": "capro-token-client/1.0"}
                )
    
            # We return the body regardless of status, as requested.
            # If you prefer to fail on non-2xx, uncomment the two lines below.
            # if not resp.ok:
            #     raise RuntimeError(f"Server returned HTTP {resp.status_code}: {resp.text.strip()}")
    
            return (resp.text or "").strip()
    
        except (Timeout, SSLError) as e:
            raise RuntimeError(f"network_or_ssl_error: {e}") from e
        except RequestException as e:
            # Covers connection errors, invalid URLs, etc.
            raise RuntimeError(f"request_error: {e}") from e



def install_pro():

    """
    Interactive installer for ChannelAttribution Pro (Python wheel).

    This helper:
    - Prompts you for a token; if you type an email instead, it requests a token
      for that address and exits (so you can rerun with the token).
    - Detects your OS/arch/Python, asks the build service for the correct wheel via HTTPS POST,
      resolves the final wheel URL, and installs it with ``python -m pip``.
    - Sends exactly one final outcome notification to ``build_check_email.php`` (SUCCESS/ERROR)
      with a short, client-trimmed info blob.

    Notes
    -----
    User interaction:
    - On start, it asks: "Enter your ChannelAttributionPro token. If you don't have one,
      enter your work/university email to request it:"
    - If you enter a valid email, the function triggers a token email and returns immediately.
    - If you enter a non-empty token, the installation flow proceeds.
    - If you enter nothing, the function prints a message and returns.

    Environment detection:
    - OS name: manylinux (Linux), macOS (macOS), Windows (Windows)
    - OS version: Linux "2014" (ManyLinux2014), macOS "13" (amd64) or "15" (arm64), Windows "11"
    - Architecture: amd64 (x86_64) or arm64 (aarch64)
    - Python: running interpreter’s major.minor (e.g., 3.11)

    Network behavior:
    - Builder request: HTTPS POST to ``https://app.channelattribution.io/genpkg/genpkg.php``
      with ``{os, os_vers, arch, lang=python, lang_vers, replace=0, uctr=0, token}``.
    - Response handling:
      - 200 or 409 with JSON containing ``"pkg"``: direct wheel URL or a directory URL.
      - If a directory is returned, the latest file is selected (prefers ``*.whl``).
      - 401 or JSON error mentioning "invalid token": prints "Token non valid or expired..." and returns.
    - Installation: runs ``python -m pip install --prefer-binary`` on the resolved wheel URL.
    - Final notification (always exactly once): POSTs to
      ``https://app.channelattribution.io/genpkg/build_check_email.php`` with fields
      ``{token, action="SUCCESS"|"ERROR", info="<=8KiB compact JSON-like"}``.

    Packages and proxies:
    - Auto-installs ``requests`` (and ``distro`` on Linux) into the current interpreter.
      If not in a virtualenv, sets ``PIP_BREAK_SYSTEM_PACKAGES=1`` to respect Debian/Ubuntu PEP 668.
    - Honors standard system/proxy environment variables.

    Output and errors:
    - Prints progress and a short system report if something goes wrong (OS, distro, Python, compiler).
      Support message includes ``info@channelattribution.io``.
    - On success it prints:

      ::

          *** Package installed. Restart the session and try to import it with: import ChannelAttributionPro

    - Returns ``None``. Most error conditions are handled by printing and returning.
    - May raise ``ValueError`` only if an email-like input is given but fails basic syntax validation.

    Security notes:
    - Sends your token to the builder and in the final notification.
    - The ``info`` field is trimmed client-side (~8 KiB) and HTML-escaped server-side before emailing.
    - No personal data is sent beyond the token and generic environment traits.

    Examples
    --------
    >>> from ChannelAttribution import install_pro
    >>> install_pro()  # follow the prompt: paste token, or enter email to request one
    """

    import sys, os, platform, json, subprocess, shutil, re
    from html.parser import HTMLParser
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
    from urllib.parse import urlencode, urljoin
    import importlib.util, sysconfig
    from getpass import getpass

    # ---------- helpers: stdlib detection + ensure_package ----------
    STDLIB_DIR = sysconfig.get_paths().get("stdlib", "")

    def is_stdlib(mod_name: str) -> bool:
        spec = importlib.util.find_spec(mod_name)
        if spec is None or spec.origin is None:
            return False
        return spec.origin == "built-in" or spec.origin.startswith(STDLIB_DIR)

    def ensure_package(import_name: str, pip_name: str | None = None, version: str | None = None):
        if is_stdlib(import_name):
            return
        try:
            importlib.util.find_spec(import_name) or __import__(import_name)
            return
        except Exception:
            pass
        pkg = pip_name or import_name
        if version:
            pkg = f"{pkg}{version}"
        env = dict(os.environ)
        in_venv = (sys.prefix != sys.base_prefix)
        if not in_venv:
            env.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", pkg],
            env=env,
        )

    def _read_secret(prompt: str = "Enter value: ") -> str:
        """Visible prompt (works with/without TTY)."""
        try:
            # Standard interactive prompt (visible)
            return input(prompt).strip()
        except EOFError:
            # Non-interactive / piped input fallback
            print(prompt, end="", flush=True)
            return sys.stdin.readline().rstrip("\n")

    _email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    def _is_valid_email(s: str) -> bool:
        return isinstance(s, str) and bool(_email_re.match(s.strip()))

    # ---------- early reachability check for app.channelattribution.io ----------
    def _can_reach_app(timeout: int = 60) -> bool:
        try:
            # Simple HTTPS GET to the root; proxies/env vars are honored by urllib
            with urlopen("https://app.channelattribution.io", timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                return 200 <= status < 400
        except Exception:
            return False

    if not _can_reach_app():
        print(
            "It seems that app.channelattribution.io cannot be reached from this environment.\n"
            "To install ChannelAttribution Pro you need to reach app.channelattribution.io.\n"
            "If you can't reach it, please write us at info@channelattribution.io."
        )
        return

    # Only third-party:
    ensure_package("requests")
    # 'distro' is useful only on Linux
    if sys.platform.startswith("linux"):
        ensure_package("distro")

    import requests
    from requests.exceptions import RequestException, Timeout, SSLError
    distro = None
    if sys.platform.startswith("linux"):
        try:
            import distro as _d  # type: ignore
            distro = _d
        except Exception:
            distro = None

    # ---------- notifier (called exactly once at the end) ----------
    def notify_package_request(
        token: str,
        action: str,
        info: str,
        endpoint: str = "https://app.channelattribution.io/genpkg/build_check_email.php",
        timeout: int = 60,
        verify_ssl: bool = True,
    ) -> str:
        if not token:
            return "missing_token_param"
        data = {"token": token, "action": action, "info": info}
        try:
            r = requests.post(
                endpoint,
                data=data,
                timeout=timeout,
                verify=verify_ssl,
                allow_redirects=True,
                headers={
                    "User-Agent": "capro-build-check/1.1",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            return (r.text or "").strip()
        except (Timeout, SSLError) as e:
            return f"network_or_ssl_error: {e}"
        except RequestException as e:
            return f"request_error: {e}"

    # ---------- small HTTP helper for builder (POST-only now) ----------
    UA = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    HEADERS = {
        "User-Agent": UA,
        "Accept": "application/json,text/html;q=0.8,*/*;q=0.5",
        "Accept-Language": "en-US,en;q=0.7",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    def http_post_form(url: str, form: dict, timeout: int = 300):
        data = urlencode(form).encode("utf-8")
        req = Request(url, data=data, headers=HEADERS, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(), resp.headers
        except HTTPError as e:
            try:
                body = e.read()
            except Exception:
                body = b""
            return e.code, body, getattr(e, "headers", {})
        except URLError as e:
            return 0, str(e).encode("utf-8", "replace"), {}

    # ---------- HTML listing helper ----------
    class LinkCollector(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() == "a":
                href = dict(attrs).get("href")
                if href:
                    self.links.append(href)

    def list_dir_files(dir_url):
        # Directory listings are fetched with GET (static files)
        try:
            with urlopen(Request(dir_url, headers={"User-Agent": UA}), timeout=60) as resp:
                status = resp.status
                body = resp.read()
        except HTTPError as e:
            status = e.code
            body = e.read()
        except URLError as e:
            print(f"Listing {dir_url} network error: {e}")
            return None
        if status != 200:
            print(f"Listing {dir_url} failed with HTTP {status}")
            return None
        html = body.decode("utf-8", errors="replace")
        p = LinkCollector()
        p.feed(html)
        return [h for h in p.links if h and h not in ("/", "../") and not h.endswith("/")]

    def resolve_pkg_url(pkg_value):
        if not isinstance(pkg_value, str) or not pkg_value:
            print("Invalid 'pkg' value in response.")
            return None
        if pkg_value.lower().endswith(".whl"):
            return pkg_value
        pkg_dir = pkg_value.rstrip("/") + "/"
        files = list_dir_files(pkg_dir)
        if files is None:
            return None
        wheels = [f for f in files if f.endswith(".whl")]
        chosen = (sorted(wheels) or sorted(files) or [None])[-1]
        if not chosen:
            print(f"No files found at {pkg_dir}")
            return None
        return urljoin(pkg_dir, chosen)

    def pip_install(url, extra_args=None):
        if not url:
            print("No package URL to install.")
            return False
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "--prefer-binary",
            url,
        ]
        if extra_args:
            cmd.extend(extra_args)
        print("Installing with:", " ".join(cmd))
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"pip failed with exit code {proc.returncode}")
            return False
        return True

    # Prompt: token or email
    msg = (
        "Enter your ChannelAttributionPro token. "
        "If you don't have one, enter your work/university email to request it: "
    )
    token = _read_secret(msg).strip()

    # If it looks like an email, trigger the token request and exit
    if "@" in token:
        email = token
        if not _is_valid_email(email):
            raise ValueError("Please enter a valid email address or a token.")
        print("Sending a token...")
        _ = __request_token_channelattributionpro(email=email)
        print(
            "*** We email the token to eligible work or university addresses - check your inbox and Spam/Junk; "
            "if you don't receive it, try a different work/university email, and if it still doesn't arrive, "
            "contact info@channelattribution.io."
        )
        return  # exit early; user will rerun with token

    if not token:
        print("A non-empty token or email is required.")
        return

    # ---------- detect environment ----------
    if sys.platform.startswith("linux"):
        os_name = "manylinux"
    elif sys.platform == "darwin":
        os_name = "macos"
    elif sys.platform in ("win32", "cygwin", "msys"):
        os_name = "windows"
    else:
        os_name = "manylinux"

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = "amd64"

    lang = "python"
    lang_vers = f"{sys.version_info.major}.{sys.version_info.minor}"

    if os_name == "macos":
        os_vers = "13" if arch == "amd64" else "15"
    elif os_name == "windows":
        os_vers = "11"
    else:
        os_vers = "2014"

    params = {
        "os": os_name,
        "os_vers": os_vers,
        "arch": arch,
        "lang": lang,
        "lang_vers": lang_vers,
        "replace": "0",
        "uctr": "0",
        "token": token,
    }

    BASE_URL = "https://app.channelattribution.io/genpkg/genpkg.php"

    # ---------- one-shot try/finally to guarantee single notification ----------
    action = "ERROR"   # will flip to SUCCESS on good path
    info_blob = ""     # what we'll send to build_check_email.php

    def get_system_info_dict():
        sysname = platform.system()
        release = platform.release()
        archi = platform.machine() or platform.processor() or "unknown"
        py_impl = platform.python_implementation()
        py_ver = platform.python_version()
        distro_str = None

        if sysname == "Linux":
            try:
                if distro:
                    name = distro.name(pretty=True) or (distro.id() or "")
                    vers = distro.version(best=True) or ""
                    distro_str = " ".join(x for x in (name, vers) if x).strip() or None
            except Exception:
                pass
            if not distro_str:
                try:
                    data = {}
                    with open(
                        "/etc/os-release",
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        for line in f:
                            line = line.strip()
                            if not line or "=" not in line or line.startswith("#"):
                                continue
                            k, v = line.split("=", 1)
                            data[k] = v.strip().strip('"').strip("'")
                    distro_str = data.get("PRETTY_NAME") or " ".join(
                        x for x in (data.get("NAME"), data.get("VERSION")) if x
                    )
                except Exception:
                    pass
        elif sysname == "Darwin":
            try:
                p = subprocess.run(
                    ["sw_vers", "-productVersion"],
                    capture_output=True,
                    text=True,
                )
                if p.returncode == 0:
                    distro_str = f"macOS {p.stdout.strip()}"
            except Exception:
                pass
        elif sysname == "Windows":
            distro_str = f"Windows {platform.release()} (build {platform.version()})"

        def _which_compiler():
            for exe in ("gcc", "cc", "clang"):
                if shutil.which(exe):
                    return exe
            return None

        def _compiler_version(exe):
            if not exe:
                return None
            try:
                p = subprocess.run(
                    [exe, "-dumpfullversion"],
                    capture_output=True,
                    text=True,
                )
                if p.returncode == 0 and p.stdout.strip():
                    return f"{exe} {p.stdout.strip()}"
            except Exception:
                pass
            try:
                p = subprocess.run(
                    [exe, "--version"],
                    capture_output=True,
                    text=True,
                )
                if p.returncode == 0 and p.stdout:
                    first = p.stdout.splitlines()[0].strip()
                    m = re.search(
                        r"(gcc|clang)[^0-9]*([0-9]+(?:\.[0-9]+){0,3})",
                        first,
                        re.I,
                    )
                    return (
                        f"{m.group(1).lower()} {m.group(2)}"
                        if m
                        else first
                    )
            except Exception:
                pass
            return None

        comp = _which_compiler()
        comp_ver = _compiler_version(comp) if comp else None

        return {
            "os": sysname,
            "os_release": release,
            "architecture": archi,
            "distro": distro_str,
            "python_implementation": py_impl,
            "python_version": py_ver,
            "compiler": comp_ver or "not found",
        }

    # pretty-printed version for user output
    def get_system_info():
        return json.dumps(get_system_info_dict(), indent=2)

    try:
        print("Building the package. Estimated time: 0-30 minutes. Please wait...")

        # POST to builder (matches hardened PHP)
        status, body, headers = http_post_form(BASE_URL, params, timeout=32*60)
        text = (
            body.decode("utf-8", errors="replace")
            if isinstance(body, (bytes, bytearray))
            else str(body)
        )

        if status == 401:
            print("Token non valid or expired. Write to info@channelattribution.io.")
            action = "ERROR"
            info_blob = json.dumps(
                {
                    "reason": "invalid_token",
                    "builder_status": status,
                    "system": get_system_info_dict(),
                },
                indent=2,
            )
            return

        data = None
        try:
            data = json.loads(text)
        except Exception:
            data = None

        # defensive token check in-body
        if isinstance(data, dict):
            err = (data.get("error") or "").lower()
            stat = (data.get("status") or "").lower()
            if "invalid token" in err or (
                stat in ("fail", "error") and "token" in err
            ):
                print("Token non valid or expired. Write to info@channelattribution.io.")
                action = "ERROR"
                info_blob = json.dumps(
                    {
                        "reason": "invalid_token_in_body",
                        "builder_status": status,
                        "body": (text[:500] if text else ""),
                        "system": get_system_info_dict(),
                    },
                    indent=2,
                )
                return

        # Resolve wheel URL
        pkg_file_url = None
        ok_path = True
        if status in (200, 409) and isinstance(data, dict) and "pkg" in data:
            pkg_file_url = resolve_pkg_url(data["pkg"])
            if not pkg_file_url:
                ok_path = False
        else:
            ok_path = False
            # Friendly support message
            print("Installation failed. Send the following information:\n")
            print(get_system_info())
            print("\nto info@channelattribution.io.")
            action = "ERROR"
            info_blob = json.dumps(
                {
                    "result": "builder_unexpected_response",
                    "builder_status": status,
                    "body": (text[:500] if text else ""),
                    "system": get_system_info_dict(),
                },
                indent=2,
            )

        if ok_path and pkg_file_url:
            ok = pip_install(pkg_file_url)
            if ok:
                print(
                    "*** Package installed. Restart the session and try to import it with: import ChannelAttributionPro"
                )
                action = "SUCCESS"
                info_blob = json.dumps(
                    {
                        "result": "installed",
                        "wheel": pkg_file_url,
                        "system": get_system_info_dict(),
                    },
                    indent=2,
                )
                return
            else:
                print("Installation failed. Send the following information:\n")
                print(get_system_info())
                print("\nto info@channelattribution.io.")
                action = "ERROR"
                info_blob = json.dumps(
                    {
                        "result": "pip_failed",
                        "wheel": pkg_file_url,
                        "system": get_system_info_dict(),
                    },
                    indent=2,
                )
                return
        elif not ok_path:
            # already printed support message above and set info_blob/action
            return

    finally:
        # ONE final notification (success or error). Keep it short on the wire.
        try:
            info_to_send = (info_blob or "")
            if len(info_to_send) > 8192:
                info_to_send = info_to_send[:8192] + "...(truncated)"
            _ = notify_package_request(token, action, info_to_send)
        except Exception:
            pass
