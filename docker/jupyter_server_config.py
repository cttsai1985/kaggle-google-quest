from pathlib import Path

c = get_config()  # noqa: F821
secret = Path("/run/secrets/jupyter_token").read_text().strip()
if not secret:
    raise ValueError("A nonempty Jupyter token is required")
c.IdentityProvider.token = secret
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.root_dir = "/workspace"
