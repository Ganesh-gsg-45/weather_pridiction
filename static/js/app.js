/**
 * AccuWeather Clone — Frontend JavaScript
 * Handles: city search autocomplete, geolocation, hero background updates
 */

'use strict';

// ── City Search Autocomplete ─────────────────────────────────────────────────
function initSearch(inputId, dropdownId) {
  const input    = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);
  if (!input || !dropdown) return;

  let debounceTimer = null;

  input.addEventListener('input', () => {
    const q = input.value.trim();
    clearTimeout(debounceTimer);
    if (q.length < 2) {
      dropdown.classList.remove('open');
      dropdown.innerHTML = '';
      return;
    }
    debounceTimer = setTimeout(() => fetchSuggestions(q, dropdown), 300);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const q = input.value.trim();
      if (q) navigateToCity(q);
    }
    if (e.key === 'Escape') {
      dropdown.classList.remove('open');
    }
  });

  // Close dropdown on outside click
  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.remove('open');
    }
  });
}

async function fetchSuggestions(query, dropdown) {
  try {
    const res  = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();

    if (!data.length) {
      dropdown.classList.remove('open');
      return;
    }

    dropdown.innerHTML = data.map(item => `
      <div class="search-dropdown-item"
           onclick="navigateToLatLon(${item.lat}, ${item.lon}, '${escHtml(item.name)}')"
           role="option" tabindex="0"
           onkeydown="if(event.key==='Enter') navigateToLatLon(${item.lat}, ${item.lon}, '${escHtml(item.name)}')">
        <span class="loc-icon">📍</span>
        <span class="loc-name">${escHtml(item.name)}${item.state ? ', ' + escHtml(item.state) : ''}</span>
        <span class="loc-country">${escHtml(item.country)}</span>
      </div>
    `).join('');

    dropdown.classList.add('open');
  } catch (err) {
    console.warn('Search error:', err);
  }
}

function navigateToCity(city) {
  window.location.href = `/weather?city=${encodeURIComponent(city)}`;
}

function navigateToLatLon(lat, lon, name) {
  window.location.href = `/weather?city=${encodeURIComponent(name)}&lat=${lat}&lon=${lon}`;
}

function escHtml(str) {
  return String(str).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}

// ── Geolocation ──────────────────────────────────────────────────────────────
function initGeolocation() {
  const btn = document.getElementById('geo-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.');
      return;
    }
    btn.textContent = '⏳ Locating…';
    btn.disabled = true;

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude: lat, longitude: lon } = pos.coords;
        window.location.href = `/weather?lat=${lat}&lon=${lon}`;
      },
      () => {
        btn.textContent = '📍 Location';
        btn.disabled = false;
        alert('Unable to retrieve your location. Please try searching manually.');
      },
      { timeout: 10000 }
    );
  });
}

// ── Dynamic Hero Background ──────────────────────────────────────────────────
function setHeroByTime() {
  const sky = document.getElementById('hero-sky');
  if (!sky) return;

  const h = new Date().getHours();
  let gradient;

  if (h >= 5 && h < 8) {
    // Dawn
    gradient = 'linear-gradient(165deg,#1a0a00 0%,#8b3a00 20%,#d4631a 40%,#f0944a 55%,#f6c27a 65%,#f0d890 72%,#a8c5e8 82%,#5588cc 92%,#2a4a7a 100%)';
  } else if (h >= 8 && h < 17) {
    // Daytime
    gradient = 'linear-gradient(165deg,#1a4f8a 0%,#2563eb 30%,#3b82f6 55%,#7dd3fc 75%,#bae6fd 90%,#e0f2fe 100%)';
  } else if (h >= 17 && h < 20) {
    // Dusk/Sunset
    gradient = 'linear-gradient(165deg,#1a0a00 0%,#7c2d12 15%,#c2410c 30%,#ea580c 45%,#fb923c 60%,#fbbf24 70%,#fde68a 78%,#fed7aa 85%,#7c3aed 95%,#1e1b4b 100%)';
  } else {
    // Night
    gradient = 'linear-gradient(165deg,#000010 0%,#0a0a2e 20%,#0f172a 50%,#1e293b 75%,#0f2027 100%)';
  }
  sky.style.background = gradient;
}

// ── Scroll reveal animation ──────────────────────────────────────────────────
function initScrollReveal() {
  const observer = new IntersectionObserver(
    (entries) => entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.animationPlayState = 'running';
        observer.unobserve(e.target);
      }
    }),
    { threshold: 0.1 }
  );
  document.querySelectorAll('.fade-in-up').forEach(el => {
    el.style.animationPlayState = 'paused';
    observer.observe(el);
  });
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initSearch('search-input', 'search-dropdown');
  initSearch('nav-search',   'nav-search-dropdown');
  initGeolocation();
  setHeroByTime();
  initScrollReveal();

  // Save city to recent locations in localStorage when visiting weather page
  const cityEl = document.querySelector('.city-name');
  if (cityEl) {
    const city = cityEl.textContent.split(',')[0].trim();
    saveRecentCity(city);
  }
});

// ── Recent Cities ────────────────────────────────────────────────────────────
function saveRecentCity(city) {
  try {
    let recent = JSON.parse(localStorage.getItem('accu_recent') || '[]');
    recent = [city, ...recent.filter(c => c !== city)].slice(0, 5);
    localStorage.setItem('accu_recent', JSON.stringify(recent));
  } catch (_) {}
}
