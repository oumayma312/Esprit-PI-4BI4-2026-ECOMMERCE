import { Component, OnDestroy, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { Subscription } from 'rxjs';
import { AuthService, UserAccount } from '../services/auth.service';

interface DashboardViewModel {
  label: string;
  title: string;
  description: string;
  iframeSrc: string;
}

const POWER_BI_REPORT_ID = 'b1a88ce6-8343-44b3-b9d3-cb976a15612c';
const POWER_BI_TENANT_ID = '604f1a96-cbe8-43f8-abbf-f8eaf5d85730';

function buildPowerBiEmbedUrl(pageName: string) {
  return `https://app.powerbi.com/reportEmbed?reportId=${POWER_BI_REPORT_ID}&autoAuth=true&ctid=${POWER_BI_TENANT_ID}&pageName=${pageName}&navContentPaneEnabled=false&filterPaneEnabled=false`;
}

const DASHBOARD_CONTENT: Record<string, DashboardViewModel> = {
  ceo: {
    label: 'CEO dashboard',
    title: 'Executive oversight for Project Story',
    description:
      'Review the project Power BI report with a focused, executive-friendly layout that keeps the key performance story front and center.',
    iframeSrc: buildPowerBiEmbedUrl('31efd327976bd0a20bb6')
  },
  sales: {
    label: 'Sales dashboard',
    title: 'Commercial performance for Project Story',
    description:
      'Open the dedicated sales view in a streamlined interface designed for pipeline visibility, momentum tracking, and faster decision-making.',
    iframeSrc: buildPowerBiEmbedUrl('9ea9f50d9813a6eb464c')
  },
  finance: {
    label: 'Finance dashboard',
    title: 'Financial reading of Project Story',
    description:
      'Use the finance report in a clean shell optimized for readability, margin review, and fast access to the numbers that matter.',
    iframeSrc: buildPowerBiEmbedUrl('bc4646f6dd1d4260038a')
  }
};

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit, OnDestroy {
  role = '';
  iframeHtml: SafeHtml = '' as any;
  viewModel: DashboardViewModel = DASHBOARD_CONTENT['ceo'];
  currentUser: UserAccount | null = null;

  private routeSub?: Subscription;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private sanitizer: DomSanitizer,
    private auth: AuthService
  ) {}

  ngOnInit(): void {
    this.currentUser = this.auth.current();

    if (!this.currentUser) {
      this.router.navigate(['/login']);
      return;
    }

    this.routeSub = this.route.paramMap.subscribe(params => {
      const requestedRole = (params.get('role') || '').toLowerCase();
      this.configureDashboard(requestedRole);
    });
  }

  ngOnDestroy(): void {
    this.routeSub?.unsubscribe();
  }

  private configureDashboard(requestedRole: string) {
    const requestedDashboard = DASHBOARD_CONTENT[requestedRole];

    if (!requestedDashboard) {
      this.redirectToAllowedDashboard();
      return;
    }

    if (!this.currentUser || !this.auth.allowedDashboardRoles(this.currentUser.role).includes(requestedRole)) {
      this.redirectToAllowedDashboard();
      return;
    }

    this.role = requestedRole;
    this.viewModel = requestedDashboard;
    this.iframeHtml = this.sanitizer.bypassSecurityTrustHtml(
      `<iframe title="Story Power BI dashboard" width="1140" height="650" src="${this.viewModel.iframeSrc}" frameborder="0" allowFullScreen="true"></iframe>`
    );
  }

  private redirectToAllowedDashboard() {
    if (!this.currentUser) {
      this.router.navigate(['/login']);
      return;
    }

    const fallbackRole = this.auth.allowedDashboardRoles(this.currentUser.role)[0] || this.currentUser.role;
    this.router.navigate(['/dashboard', fallbackRole]);
  }
}
