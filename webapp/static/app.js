(() => {
  const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const qs = (sel, el = document) => el.querySelector(sel);
  const qsa = (sel, el = document) => Array.from(el.querySelectorAll(sel));

  // Navbar toggle (mobile)
  const toggleBtn = qs('[data-nav-toggle]');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      document.body.classList.toggle('nav-open');
    });
  }

  // Close navbar menu after navigation on mobile
  qsa('a[data-nav]').forEach((a) => {
    a.addEventListener('click', (e) => {
      document.body.classList.remove('nav-open');

      // Page transition (only for normal left-click navigation)
      const isModified = e.metaKey || e.ctrlKey || e.shiftKey || e.altKey;
      if (prefersReduced || isModified || e.button !== 0) return;
      const href = a.getAttribute('href');
      if (!href || href.startsWith('http') || href.startsWith('#')) return;

      e.preventDefault();
      document.documentElement.classList.add('is-leaving');
      window.setTimeout(() => {
        window.location.href = href;
      }, 140);
    });
  });

  // Tabs
  qsa('[data-tabs]').forEach((tabsEl) => {
    const tabs = qsa('[data-tab]', tabsEl);
    const panels = qsa('[data-tabpanel]');

    const setActive = (key) => {
      tabs.forEach((t) => t.classList.toggle('is-active', t.dataset.tab === key));
      panels.forEach((p) => {
        const isMatch = p.dataset.tabpanel === key;
        p.hidden = !isMatch;
      });

      const activePanel = panels.find((p) => p.dataset.tabpanel === key);
      if (activePanel) {
        renderPlotly(activePanel);
      }
    };

    tabs.forEach((t) => {
      t.addEventListener('click', () => setActive(t.dataset.tab));
    });

    // Initialize
    const active = tabs.find((t) => t.classList.contains('is-active'));
    if (active) setActive(active.dataset.tab);
  });

  function renderPlotly(root = document) {
    if (typeof window.Plotly === 'undefined') return;

    qsa('[data-plotly]', root).forEach((card) => {
      if (card.dataset.rendered === '1') return;
      const plotEl = qs('.plotly', card);
      const jsonEl = qs('.plotly-json', card);
      if (!plotEl || !jsonEl) return;

      try {
        const spec = JSON.parse(jsonEl.textContent || '{}');
        const data = spec.data || spec.figure?.data || [];
        const layout = spec.layout || spec.figure?.layout || {};
        const config = Object.assign({ responsive: true, displaylogo: false }, spec.config || {});

        window.Plotly.react(plotEl, data, layout, config);
        card.dataset.rendered = '1';
      } catch (e) {
        // Ignore rendering errors (keep page usable)
        card.dataset.rendered = '1';
      }
    });
  }

  // First paint init (only render visible panels)
  qsa('[data-tabpanel]').forEach((p) => {
    if (!p.hidden) renderPlotly(p);
  });

  // Page enter animation
  if (!prefersReduced) {
    document.documentElement.classList.add('js');
    window.requestAnimationFrame(() => {
      document.documentElement.classList.add('is-loaded');
    });
  }
})();
