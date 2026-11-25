# Fax fixture

`std-g3.tif`: five lines of statement text (amounts, dates, a check number) as a standard-mode fax: CCITT Group 3, 1-bit, 1728 dots across,
200 x 100 dots per inch. Drawn with Pillow's built-in font at 200 x 200, every second row dropped by box averaging, then thresholded. Made by
`tools/make-fax-fixture.py`; the text is the list `LINES` in that script. CC0.
