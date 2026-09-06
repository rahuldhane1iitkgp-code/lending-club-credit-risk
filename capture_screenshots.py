"""
Capture README screenshots from the live Streamlit app.

Shoots two views: the application form, and the result panel with the decision and
SHAP drivers. Before shooting it asserts the footer carries the current model's
metrics, so a stale Streamlit cache cannot silently produce screenshots that
contradict the README.

Usage: python capture_screenshots.py
"""
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = "http://localhost:8502"
OUT = Path("D:/Lending/docs")
OUT.mkdir(exist_ok=True)

# The retrained model's figures, as printed by save_deployment_artifacts.py
EXPECT = {"PR-AUC: 0.392": "test PR-AUC", "0.699": "test ROC-AUC",
          "36%": "approval rate", "$25.2M": "net profit"}

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1180, "height": 2600},
                            device_scale_factor=2)
    print("loading app...", flush=True)
    page.goto(APP, wait_until="networkidle", timeout=120_000)
    page.wait_for_selector("text=Assess Risk", timeout=120_000)
    page.wait_for_timeout(3000)

    footer = page.inner_text("body")
    missing = [label for frag, label in EXPECT.items() if frag not in footer]
    if missing:
        print("\nSTALE APP - the footer does not carry the retrained model's numbers.", flush=True)
        print("missing:", ", ".join(missing), flush=True)
        m = re.search(r"Model: XGBoost.*", footer, re.S)
        print("footer says:", (m.group(0)[:400] if m else "not found"), flush=True)
        print("\nReboot the app from the Streamlit Cloud menu to clear "
              "@st.cache_resource, then re-run.", flush=True)
        browser.close()
        sys.exit(1)
    print("footer confirms the retrained model is live", flush=True)

    # The viewport is deliberately taller than the app so Streamlit's inner scroll
    # container renders everything at once; crop back to the form itself so the README
    # image is not mostly blank.
    btn = page.locator("text=Assess Risk").first.bounding_box()
    form_h = btn["y"] + btn["height"] + 40
    page.screenshot(path=str(OUT / "app_form.png"),
                    clip={"x": 0, "y": 0, "width": 1180, "height": form_h})
    print(f"wrote app_form.png (height {form_h:.0f})", flush=True)

    page.click("text=Assess Risk")
    page.wait_for_selector("text=Why this prediction", timeout=120_000)
    page.wait_for_timeout(2500)

    # Crop from the "Result" heading down through the SHAP driver list, which is the
    # part worth showing - a decision without its reasons is just a number.
    result = page.locator("text=Result").first
    result.scroll_into_view_if_needed()
    page.wait_for_timeout(800)
    # bounding_box() is viewport-relative but a clip on a full-page screenshot is
    # document-relative, so the scroll offset has to be added back in - otherwise the
    # SHAP list, which sits below the fold, gets cut off.
    scroll_y = page.evaluate("window.scrollY")
    doc_h = page.evaluate("document.documentElement.scrollHeight")
    top = result.bounding_box()
    y0 = max(top["y"] + scroll_y - 25, 0)
    # Run to the page bottom rather than locating the last SHAP row - the driver list
    # is the final block before the footer, so this is both simpler and robust to the
    # number of rows shown.
    height = min(840, doc_h - y0)
    page.screenshot(path=str(OUT / "app_result.png"), full_page=True,
                    clip={"x": max(top["x"] - 45, 0), "y": y0,
                          "width": 1050, "height": height})
    print(f"   result clip: y0={y0:.0f} height={height:.0f} (doc {doc_h})", flush=True)
    print("wrote app_result.png", flush=True)

    browser.close()
print("\ndone")
