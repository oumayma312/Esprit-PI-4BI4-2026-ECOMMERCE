import { Component, OnDestroy, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { finalize, Subscription } from 'rxjs';

import { AuthService, UserAccount } from '../services/auth.service';
import { DecisionItem, IaService } from '../services/ia.service';

interface DashboardProfile {
  role: string;
  title: string;
  subtitle: string;
  description: string;
  accent: string;
  surface: string;
}

const ROLE_LABEL_MAP: Record<string, string> = {
  ceo: 'CEO',
  sales: 'Sales',
  finance: 'Finance',
};

const DASHBOARD_PROFILES: Record<string, DashboardProfile> = {
  ceo: {
    role: 'CEO',
    title: 'Decision Support Command Center',
    subtitle: 'Strategic signals synthesized from KPIs, models, and data context.',
    description: 'High-level actions that balance growth, risk, and operational cadence.',
    accent: '#2d7ff9',
    surface: 'rgba(45, 127, 249, 0.12)',
  },
  sales: {
    role: 'Sales',
    title: 'Commercial Decision Support',
    subtitle: 'Campaign timing, competitor pressure, and demand signals in one queue.',
    description: 'Actions to improve conversion, retention, and promotion efficiency.',
    accent: '#f59e0b',
    surface: 'rgba(245, 158, 11, 0.16)',
  },
  finance: {
    role: 'Finance',
    title: 'Finance Decision Control',
    subtitle: 'Margin, cost, and ROI watchpoints surfaced as actionable decisions.',
    description: 'Budget and margin actions that protect cash and profitability.',
    accent: '#10b981',
    surface: 'rgba(16, 185, 129, 0.14)',
  },
};

@Component({
  selector: 'app-decision-support',
  templateUrl: './decision-support.component.html',
  styleUrls: ['./decision-support.component.css'],
})
export class DecisionSupportComponent implements OnInit, OnDestroy {
  currentUser: UserAccount | null = null;
  profile: DashboardProfile | null = null;

  decisions: DecisionItem[] = [];
  loading = false;
  error = '';
  unreadOnly = false;

  lastUpdatedAt: Date | null = null;
  currentPage = 1;
  pageSize = 8;

  private readonly subscriptions = new Subscription();
  private readonly readingIds = new Set<number>();

  constructor(
    private readonly auth: AuthService,
    private readonly router: Router,
    private readonly ia: IaService,
  ) {}

  ngOnInit(): void {
    this.currentUser = this.auth.current();

    if (!this.currentUser) {
      this.router.navigate(['/login']);
      return;
    }

    this.profile = DASHBOARD_PROFILES[this.currentUser.role] ?? DASHBOARD_PROFILES['ceo'];
    this.refresh();
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
  }

  get paginatedDecisions(): DecisionItem[] {
    const start = (this.currentPage - 1) * this.pageSize;
    return this.decisions.slice(start, start + this.pageSize);
  }

  get totalPages(): number {
    return Math.max(1, Math.ceil(this.decisions.length / this.pageSize));
  }

  get authLabel(): string {
    if (!this.currentUser) {
      return 'Not signed in';
    }
    return `Signed in as ${this.currentUser.email} (${this.currentUser.role.toUpperCase()})`;
  }

  get lastUpdatedLabel(): string {
    if (!this.lastUpdatedAt) {
      return 'Last updated: not yet';
    }
    return `Last updated: ${this.lastUpdatedAt.toLocaleString()}`;
  }

  get pendingCount(): number {
    return this.decisions.filter((item) => item.statut !== 'lu').length;
  }

  refresh(): void {
    if (!this.currentUser) {
      return;
    }

    this.loading = true;
    this.error = '';

    const roleLabel = ROLE_LABEL_MAP[this.currentUser.role] ?? 'CEO';

    const sub = this.ia
      .getDecisions(roleLabel, { unreadOnly: this.unreadOnly, limit: 50 })
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (items) => {
          this.decisions = items;
          this.lastUpdatedAt = new Date();
          this.currentPage = 1;
        },
        error: (err) => {
          this.error = this.formatError(err);
        },
      });

    this.subscriptions.add(sub);
  }

  toggleUnreadOnly(event: Event): void {
    this.unreadOnly = (event.target as HTMLInputElement).checked;
    this.refresh();
  }

  goToPage(page: number): void {
    const nextPage = Math.max(1, Math.min(this.totalPages, page));
    this.currentPage = nextPage;
  }

  priorityCount(level: string): number {
    return this.decisions.filter((item) => item.urgence === level).length;
  }

  urgencyLabel(level: string): string {
    const normalized = level?.toUpperCase();
    if (normalized === 'HAUTE') {
      return 'High';
    }
    if (normalized === 'MOYENNE') {
      return 'Medium';
    }
    if (normalized === 'STABLE') {
      return 'Stable';
    }
    return normalized || 'Unknown';
  }

  statusLabel(status: string): string {
    if (status === 'lu') {
      return 'Read';
    }
    return 'Pending';
  }

  markRead(decision: DecisionItem): void {
    if (decision.statut === 'lu' || this.readingIds.has(decision.id)) {
      return;
    }

    this.readingIds.add(decision.id);

    const sub = this.ia
      .markDecisionRead(decision.id)
      .pipe(finalize(() => this.readingIds.delete(decision.id)))
      .subscribe({
        next: () => {
          decision.statut = 'lu';
          if (this.unreadOnly) {
            this.decisions = this.decisions.filter((item) => item.id !== decision.id);
            this.currentPage = 1;
          }
        },
        error: (err) => {
          this.error = this.formatError(err);
        },
      });

    this.subscriptions.add(sub);
  }

  isReading(decisionId: number): boolean {
    return this.readingIds.has(decisionId);
  }

  trackByDecisionId(_: number, decision: DecisionItem): number {
    return decision.id;
  }

  logout(): void {
    this.auth.logout();
  }

  private formatError(error: unknown): string {
    if (typeof error === 'string') {
      return error;
    }

    if (error && typeof error === 'object' && 'message' in error) {
      return String((error as { message?: string }).message || 'Server error');
    }

    return 'Unable to load decisions right now.';
  }
}
