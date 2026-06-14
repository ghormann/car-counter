# Image Viewer Design

**Date:** 2026-06-14
**Branch:** website

## Summary

Replace the current behavior of opening a clicked thumbnail in a new browser tab with an in-page image viewer. The viewer occupies the full page, provides prev/next navigation within the current page's image list, and returns to the grid on request.

---

## Behavior Change

**Before:** Clicking a thumbnail calls `window.open(imageUrl)`, opening the full-size image in a new tab.

**After:** Clicking a thumbnail switches the page to a viewer state showing the full-size image with navigation controls.

---

## View States

The SPA has two mutually exclusive states:

- `grid` — existing filter bar, thumbnail grid, pagination controls
- `viewer` — full-page image viewer with thin control bar

A `showView(state)` function toggles visibility between the two states by showing/hiding their container `<div>` elements. No routing, no URL changes, no backend changes.

---

## Viewer Layout

```
┌─────────────────────────────────────────────────────┐
│ ←  ‹  ›   2026-06-14 12:34:56  [startup]           │  ~40px bar
├─────────────────────────────────────────────────────┤
│                                                     │
│                  [full-size image]                  │
│               object-fit: contain                   │
│               width: 100%, height: 100%             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Top bar (left to right):**
- Back arrow icon `←` — returns to grid view
- Prev icon `‹` — navigate to previous image
- Next icon `›` — navigate to next image
- Timestamp — human-readable (`YYYY-MM-DD HH:MM:SS`)
- "startup" badge — shown if filename begins with `startup_`

Icons are small (~20px), icon-only (no text labels). Bar height ~40px.

**Image area:**
- Fills remaining viewport height (`calc(100vh - 40px)`)
- `<img>` with `object-fit: contain`, `width: 100%`, `height: 100%`
- Background: black, to contrast the image

---

## Navigation

- `viewerIndex` tracks the current position in the existing `images` array (the already-loaded, sorted, filtered list for the current page — up to 50 images)
- Prev decrements, Next increments; both clamp at array boundaries (no wrapping)
- Prev/Next are disabled (visually) when at the first/last image
- Keyboard: left arrow = prev, right arrow = next (listener active only in viewer state)

---

## State Preserved on Back

Clicking the back arrow:
- Calls `showView('grid')`
- Does not reload or reset anything — grid retains its current page, filters, and scroll position

---

## Files Changed

| File | Change |
|---|---|
| `website/static/app.js` | Add `viewerIndex`, `openViewer(index)`, `showViewer(index)`, `showView()`, keyboard listener; update thumbnail click handler |
| `website/static/index.html` | Add viewer `<div>` with bar and `<img>` |
| `website/static/style.css` | Styles for viewer container, top bar, icons, image area |
| `docs/frontend_requirements.md` | Update thumbnail click behavior description |

---

## Out of Scope

- Crossing page boundaries during prev/next navigation
- Sharing/bookmarking a viewer URL
- Swipe gestures
- Image zoom
