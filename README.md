# Passport Photo Proper v14

## New in v12

Sidebar now has Canvas Mode:

- ON: all uploaded single photos and their copy counts are placed into one shared A4 canvas sequence.
  - When the first A4 page is full, a second A4 page is created automatically in the same PDF.
- OFF: each uploaded photo creates its own separate PDF/PSD canvas as before.

## Existing defaults

- Every uploaded photo default copy count: 4
- Default single photo orientation: Landscape - rotate 90 degrees clockwise
- Auto photo enhancement default: OFF

## Outputs

- PDF
- PSD flattened first A4 page
- PNG preview/first canvas
- Batch ZIP

## Note

When combined canvas mode is ON, PDF can contain multiple pages.
PSD is flattened and contains the first A4 page.


## Fix in v13

Canvas Mode now applies to all output types:
- ON: single photos + pair/combined photos all go into one shared A4 canvas sequence.
- OFF: every output remains separate, as before.

Mixed-size items are placed left-to-right until the A4 page is full, then a new A4 page is added.


## Fix in v14

Combined canvas mode now enforces the selected per-row count.
For example, if Landscape mode is selected and per row is 4, the output canvas places 4 photos per row.
The app auto-adjusts horizontal gap/margin to fit the selected row count without resizing photos.
# passport-photo-processor
