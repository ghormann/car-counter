const selCamera = document.getElementById('sel-camera');
const selYear   = document.getElementById('sel-year');
const selMonth  = document.getElementById('sel-month');
const selDay    = document.getElementById('sel-day');
const selSort   = document.getElementById('sel-sort');
const grid      = document.getElementById('grid');
const btnPrev   = document.getElementById('btn-prev');
const btnNext   = document.getElementById('btn-next');
const pageInd   = document.getElementById('page-indicator');

let currentPage = 1;

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
  return r.json();
}

function populate(sel, values, selectedValue) {
  sel.innerHTML = '';
  values.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    if (v === selectedValue) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function loadCameras() {
  try {
    const cameras = await fetchJSON('/api/cameras');
    if (!cameras.length) return;
    populate(selCamera, cameras, cameras[0]);
    await loadYears();
  } catch (e) {
    grid.innerHTML = `<p style="color:#e74c3c;padding:16px">Error loading cameras: ${e.message}</p>`;
  }
}

async function loadYears() {
  const cam = selCamera.value;
  const years = await fetchJSON(`/api/years?camera=${encodeURIComponent(cam)}`);
  // Most recent year = last in sorted list
  const latest = years[years.length - 1];
  populate(selYear, years, latest);
  await loadMonths();
}

async function loadMonths() {
  const cam  = selCamera.value;
  const year = selYear.value;
  const months = await fetchJSON(`/api/months?camera=${encodeURIComponent(cam)}&year=${encodeURIComponent(year)}`);
  const latest = months[months.length - 1];
  populate(selMonth, months, latest);
  await loadDays();
}

async function loadDays() {
  const cam   = selCamera.value;
  const year  = selYear.value;
  const month = selMonth.value;
  const days  = await fetchJSON(`/api/days?camera=${encodeURIComponent(cam)}&year=${encodeURIComponent(year)}&month=${encodeURIComponent(month)}`);
  const latest = days[days.length - 1];
  populate(selDay, days, latest);
  currentPage = 1;
  await loadImages();
}

// Event listeners — parent resets children
selCamera.addEventListener('change', loadYears);
selYear.addEventListener('change', loadMonths);
selMonth.addEventListener('change', loadDays);
selDay.addEventListener('change', () => { currentPage = 1; loadImages(); });
selSort.addEventListener('change', () => { currentPage = 1; loadImages(); });

async function loadImages() {
  const cam   = selCamera.value;
  const year  = selYear.value;
  const month = selMonth.value;
  const day   = selDay.value;
  const sort  = selSort.value;

  const url = `/api/images?camera=${encodeURIComponent(cam)}&year=${encodeURIComponent(year)}&month=${encodeURIComponent(month)}&day=${encodeURIComponent(day)}&sort=${sort}&page=${currentPage}`;
  const data = await fetchJSON(url);

  grid.innerHTML = '';
  data.images.forEach(img => {
    const a = document.createElement('a');
    a.href = img.url;
    a.target = '_blank';
    a.className = 'thumb-card';

    const image = document.createElement('img');
    image.src = img.url;
    image.alt = img.timestamp;
    image.loading = 'lazy';

    const info = document.createElement('div');
    info.className = 'thumb-info';
    info.textContent = img.timestamp;

    if (img.is_startup) {
      const badge = document.createElement('span');
      badge.className = 'badge-startup';
      badge.textContent = 'startup';
      info.appendChild(badge);
    }

    a.appendChild(image);
    a.appendChild(info);
    grid.appendChild(a);
  });

  pageInd.textContent = `Page ${data.page} of ${data.total_pages}`;
  btnPrev.disabled = data.page <= 1;
  btnNext.disabled = data.page >= data.total_pages;
}

btnPrev.addEventListener('click', () => { currentPage--; loadImages(); });
btnNext.addEventListener('click', () => { currentPage++; loadImages(); });

// Bootstrap on page load
loadCameras();
