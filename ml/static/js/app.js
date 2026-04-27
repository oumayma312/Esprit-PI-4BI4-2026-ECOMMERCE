(function () {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function setupNavTransitions() {
    if (prefersReducedMotion) {
      return;
    }

    const links = document.querySelectorAll('a[data-nav="true"]');
    links.forEach((link) => {
      link.addEventListener('click', (event) => {
        const href = link.getAttribute('href');
        if (!href || href.startsWith('#') || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
          return;
        }

        event.preventDefault();
        document.body.classList.add('is-leaving');
        window.setTimeout(() => {
          window.location.href = href;
        }, 170);
      });
    });
  }

  function setupRevealAnimation() {
    const revealNodes = document.querySelectorAll('.reveal');
    if (revealNodes.length === 0) {
      return;
    }

    if (prefersReducedMotion || !('IntersectionObserver' in window)) {
      revealNodes.forEach((node) => node.classList.add('is-visible'));
      return;
    }

    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) {
            return;
          }
          entry.target.classList.add('is-visible');
          obs.unobserve(entry.target);
        });
      },
      { threshold: 0.08 }
    );

    revealNodes.forEach((node) => observer.observe(node));
  }

  setupNavTransitions();
  setupRevealAnimation();
})();
