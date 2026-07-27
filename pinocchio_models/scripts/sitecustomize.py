"""Local Python startup patch for MeshCat's Windows server subprocess."""

import os
import ssl


def _patch_windows_ssl_store() -> None:
    if os.name != "nt":
        return

    try:
        ssl.create_default_context()
        return
    except ssl.SSLError:
        pass

    default_cafile = ssl.get_default_verify_paths().cafile

    def create_context_without_windows_store(*args, **kwargs):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        cafile = kwargs.get("cafile") or default_cafile
        capath = kwargs.get("capath")
        cadata = kwargs.get("cadata")
        if cafile or capath or cadata:
            context.load_verify_locations(
                cafile=cafile,
                capath=capath,
                cadata=cadata,
            )
        return context

    ssl.create_default_context = create_context_without_windows_store


_patch_windows_ssl_store()
