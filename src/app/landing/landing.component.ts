import {
  AfterViewInit,
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  QueryList,
  ViewChildren
} from '@angular/core';

type FeatureIcon = 'dashboard' | 'ml' | 'chat' | 'face' | 'flow' | 'security';

interface LandingFeature {
  eyebrow: string;
  title: string;
  description: string;
  icon: FeatureIcon;
  delay: string;
}

interface LandingBenefit {
  title: string;
  description: string;
  value: string;
  signal: string;
  delay: string;
}

interface ContactCard {
  title: string;
  subtitle: string;
  value: string;
}

interface JourneyStep {
  step: string;
  title: string;
  description: string;
}

interface HeroStat {
  value: string;
  label: string;
  detail: string;
}

interface SceneMetric {
  label: string;
  value: string;
  trend: string;
}

@Component({
  selector: 'app-landing',
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.css']
})
export class LandingComponent implements AfterViewInit, OnDestroy {
  @ViewChildren('revealTarget', { read: ElementRef })
  revealTargets!: QueryList<ElementRef<HTMLElement>>;

  navElevated = false;
  mobileNavOpen = false;
  motionReady = false;
  reduceMotion = false;
  currentTheme: 'dark' | 'light' = this.resolveInitialTheme();

  visitorName = '';
  visitorEmail = '';
  visitorCompany = '';
  visitorMessage = '';
  contactFeedback = '';
  contactFeedbackTone: 'success' | 'warning' = 'success';

  heroRotateX = '0deg';
  heroRotateY = '0deg';
  heroShiftX = '0px';
  heroShiftY = '0px';
  heroGlowX = '64%';
  heroGlowY = '38%';
  heroScrollShift = '0px';

  readonly heroSignals = ['Power BI dashboards', 'AI copilots', 'Secure Face ID', 'FastAPI services'];

  readonly heroStats: HeroStat[] = [
    {
      value: '3',
      label: 'decision hubs',
      detail: 'CEO, Sales and Finance aligned in one command layer'
    },
    {
      value: '5',
      label: 'workflows ML',
      detail: 'Predictions, recommendations and timing signals in one place'
    },
    {
      value: '1',
      label: 'secure journey',
      detail: 'Password and biometric sign-in within the same flow'
    }
  ];

  readonly sceneMetrics: SceneMetric[] = [
    {
      label: 'Insight velocity',
      value: 'Realtime',
      trend: 'Dashboards and signals aligned in one surface'
    },
    {
      label: 'ML layer',
      value: '5 models',
      trend: 'Classification, promotion and selling scenarios'
    },
    {
      label: 'Access trust',
      value: 'Face ID',
      trend: 'Less friction, more trust across access points'
    }
  ];

  readonly chartHeights = [42, 58, 47, 76, 62, 88, 73];
  readonly workflowNodes = ['Collect and unify', 'Predict and score', 'Activate teams'];

  readonly trustSignals = [
    'Angular frontend',
    'FastAPI backend',
    'Machine learning workflows',
    'PostgreSQL data access',
    'Power BI storytelling',
    'Biometric authentication'
  ];

  readonly storyPoints = [
    'The hero presents a true product surface with depth, lighting and a strong visual anchor.',
    'Glassmorphism cards and subtle neon halos immediately raise the perceived product quality.',
    'Scroll reveals and mouse reactions make the visit feel more alive without sacrificing smoothness.'
  ];

  readonly features: LandingFeature[] = [
    {
      eyebrow: 'Executive views',
      title: 'Business dashboards, orchestrated',
      description:
        'CEO, Sales and Finance experiences that feel connected, coherent and ready to operate.',
      icon: 'dashboard',
      delay: '80ms'
    },
    {
      eyebrow: 'ML operations',
      title: 'Visible predictions and recommendations',
      description:
        'Machine learning modules move from hidden models to clear product capabilities that feel useful and premium.',
      icon: 'ml',
      delay: '140ms'
    },
    {
      eyebrow: 'AI guidance',
      title: 'Integrated analytics assistant',
      description:
        'The chatbot becomes part of the story as a copilot for exploration, synthesis and fast answers.',
      icon: 'chat',
      delay: '200ms'
    },
    {
      eyebrow: 'Identity layer',
      title: 'Premium biometric access',
      description:
        'The sign-in flow feels more modern and trustworthy with a more elegant Face ID experience.',
      icon: 'face',
      delay: '260ms'
    },
    {
      eyebrow: 'Conversion flow',
      title: 'A clearer path to action',
      description:
        'Each section helps visitors understand, imagine the outcome and click forward without visual overload.',
      icon: 'flow',
      delay: '320ms'
    },
    {
      eyebrow: 'Enterprise trust',
      title: 'Role-based access with confidence',
      description:
        'The product communicates stronger technical credibility through a staging that highlights security and control.',
      icon: 'security',
      delay: '380ms'
    }
  ];

  readonly benefits: LandingBenefit[] = [
    {
      value: '3 decision hubs',
      signal: 'Clear narrative',
      title: 'An immediate product promise',
      description:
        'Visitors understand within seconds that the platform connects dashboards, AI and security inside one universe.',
      delay: '80ms'
    },
    {
      value: '5 ML signals',
      signal: 'Business impact',
      title: 'More tangible product modules',
      description:
        'Prediction layers become visible, readable and clearly useful for real teams and real decisions.',
      delay: '160ms'
    },
    {
      value: '1 premium flow',
      signal: 'Stronger perception',
      title: 'A more professional image',
      description:
        'The visual direction makes the experience feel like a mature startup product instead of a simple page stack.',
      delay: '240ms'
    }
  ];

  readonly journeySteps: JourneyStep[] = [
    {
      step: '01',
      title: 'Capture',
      description: 'The hero sets an immersive scene that catches attention before the detailed reading begins.'
    },
    {
      step: '02',
      title: 'Reassure',
      description: 'The next sections reveal the modules, structure and business logic behind the platform.'
    },
    {
      step: '03',
      title: 'Convert',
      description: 'The CTA, sign-in and contact flow close the loop without breaking the premium experience.'
    }
  ];

  readonly contactCards: ContactCard[] = [
    {
      title: 'Request a demo',
      subtitle: 'For a product presentation or guided walkthrough.',
      value: 'demo@projectstory.ai'
    },
    {
      title: 'Project support',
      subtitle: 'For technical, data or dashboard-related questions.',
      value: 'support@projectstory.ai'
    },
    {
      title: 'Product team',
      subtitle: 'For use cases, roadmap conversations and deployment planning.',
      value: '+216 70 000 000'
    }
  ];

  private observer?: IntersectionObserver;

  @HostListener('window:scroll')
  onWindowScroll(): void {
    const scrollY = typeof window === 'undefined' ? 0 : window.scrollY;
    this.navElevated = scrollY > 18;

    if (this.reduceMotion) {
      this.heroScrollShift = '0px';
      return;
    }

    const cappedScroll = Math.min(scrollY, 420);
    this.heroScrollShift = `${(cappedScroll * -0.14).toFixed(2)}px`;
  }

  @HostListener('window:resize')
  onWindowResize(): void {
    if (typeof window === 'undefined') {
      return;
    }

    if (window.innerWidth > 960) {
      this.mobileNavOpen = false;
      return;
    }

    this.resetHeroPointer();
  }

  ngAfterViewInit(): void {
    if (typeof window !== 'undefined' && 'matchMedia' in window) {
      this.reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    this.motionReady = !this.reduceMotion;
    this.onWindowScroll();

    if (typeof window === 'undefined' || !('IntersectionObserver' in window)) {
      this.revealTargets.forEach((target) => target.nativeElement.classList.add('is-visible'));
      return;
    }

    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            this.observer?.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.16,
        rootMargin: '0px 0px -48px 0px'
      }
    );

    this.revealTargets.forEach((target) => this.observer?.observe(target.nativeElement));
  }

  ngOnDestroy(): void {
    this.observer?.disconnect();
  }

  onHeroPointerMove(event: MouseEvent): void {
    if (this.reduceMotion) {
      return;
    }

    const container = event.currentTarget as HTMLElement | null;

    if (!container) {
      return;
    }

    const rect = container.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    const rotateX = (0.5 - y) * 14;
    const rotateY = (x - 0.5) * 18;
    const shiftX = (x - 0.5) * 30;
    const shiftY = (y - 0.5) * 20;

    this.heroRotateX = `${rotateX.toFixed(2)}deg`;
    this.heroRotateY = `${rotateY.toFixed(2)}deg`;
    this.heroShiftX = `${shiftX.toFixed(2)}px`;
    this.heroShiftY = `${shiftY.toFixed(2)}px`;
    this.heroGlowX = `${(x * 100).toFixed(2)}%`;
    this.heroGlowY = `${(y * 100).toFixed(2)}%`;
  }

  resetHeroPointer(): void {
    this.heroRotateX = '0deg';
    this.heroRotateY = '0deg';
    this.heroShiftX = '0px';
    this.heroShiftY = '0px';
    this.heroGlowX = '64%';
    this.heroGlowY = '38%';
  }

  toggleMobileNav(): void {
    this.mobileNavOpen = !this.mobileNavOpen;
  }

  toggleTheme(): void {
    this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
    this.persistTheme();
  }

  closeMobileNav(): void {
    this.mobileNavOpen = false;
  }

  submitContact(): void {
    const name = this.visitorName.trim();
    const email = this.visitorEmail.trim();
    const company = this.visitorCompany.trim();
    const message = this.visitorMessage.trim();

    if (!name || !email || !message) {
      this.contactFeedback = 'Please provide your name, email and message to continue.';
      this.contactFeedbackTone = 'warning';
      return;
    }

    const subject = encodeURIComponent(`Project Story contact - ${company || name}`);
    const body = encodeURIComponent(
      `Name: ${name}\nEmail: ${email}\nCompany: ${company || 'Not provided'}\n\nMessage:\n${message}`
    );

    window.location.href = `mailto:demo@projectstory.ai?subject=${subject}&body=${body}`;

    this.contactFeedback = 'Your email client has been prepared. We will get back to you shortly.';
    this.contactFeedbackTone = 'success';
    this.visitorName = '';
    this.visitorEmail = '';
    this.visitorCompany = '';
    this.visitorMessage = '';
  }

  private resolveInitialTheme(): 'dark' | 'light' {
    if (typeof window === 'undefined') {
      return 'dark';
    }

    const storedTheme = window.localStorage.getItem('project-story-landing-theme');

    if (storedTheme === 'light' || storedTheme === 'dark') {
      return storedTheme;
    }

    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }

  private persistTheme(): void {
    if (typeof window === 'undefined') {
      return;
    }

    window.localStorage.setItem('project-story-landing-theme', this.currentTheme);
  }
}
