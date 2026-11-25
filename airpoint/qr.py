"""A QR code as a PNG, drawn by macOS itself (Core Image): nothing to install.

    venv/bin/python -m airpoint.qr https://example.org/try/ try/qr.png
"""
import sys

import cv2
import numpy as np

QUIET = 4       # white modules around the code, as the standard asks


def make(text, path, module_px=12):
    """Write a QR code for `text` to `path`. -> (side in px, what a reader gets back from the picture)."""
    from AppKit import NSBitmapImageRep, NSPNGFileType
    from Foundation import NSString, NSUTF8StringEncoding
    from Quartz import CGAffineTransformMakeScale, CIFilter
    f = CIFilter.filterWithName_("CIQRCodeGenerator")
    f.setValue_forKey_(NSString.stringWithString_(text).dataUsingEncoding_(NSUTF8StringEncoding), "inputMessage")
    f.setValue_forKey_("M", "inputCorrectionLevel")
    image = f.outputImage().imageByApplyingTransform_(CGAffineTransformMakeScale(module_px, module_px))
    png = NSBitmapImageRep.alloc().initWithCIImage_(image).representationUsingType_properties_(NSPNGFileType, None)
    grey = cv2.imdecode(np.frombuffer(bytes(png), np.uint8), cv2.IMREAD_GRAYSCALE)
    dark = np.argwhere(grey < 128)
    (y0, x0), (y1, x1) = dark.min(axis=0), dark.max(axis=0)
    code = cv2.copyMakeBorder(grey[y0:y1 + 1, x0:x1 + 1], *[QUIET * module_px] * 4, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, code)
    return code.shape[0], read(path)


def read(path):
    """-> the text in a QR picture, or "" when none is found."""
    return cv2.QRCodeDetector().detectAndDecode(cv2.imread(path))[0]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: python -m airpoint.qr <text or link> <out.png>")
    side, back = make(argv[0], argv[1])
    if back != argv[0]:
        raise SystemExit(f"wrote {argv[1]} but it does not read back ({back!r})")
    print(f"wrote {argv[1]} ({side}x{side} px); it reads back as {back}")


if __name__ == "__main__":
    main()
