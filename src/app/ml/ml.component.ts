import { Component, DestroyRef, OnInit, inject } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { startWith } from 'rxjs';

import {
  CustomerChurnCustomerRow,
  CustomerChurnSummaryResponse,
  MlOverviewResponse,
  MlPredictionResponse,
  MlService,
} from '../services/ml.service';

type WorkflowId =
  | 'competitorClassification'
  | 'campaignSuccess'
  | 'customerChurn'
  | 'bestTimeToSell'
  | 'bestTimeToPromote'
  | 'supplierClassification';

interface WorkflowCard {
  id: WorkflowId;
  kicker: string;
  title: string;
  subtitle: string;
  shortCode: string;
  buttonLabel: string;
}

@Component({
  selector: 'app-ml',
  templateUrl: './ml.component.html',
  styleUrls: ['./ml.component.css'],
})
export class MlComponent implements OnInit {
  private readonly destroyRef = inject(DestroyRef);
  private readonly fb = inject(FormBuilder);
  private readonly mlService = inject(MlService);

  readonly workflows: WorkflowCard[] = [
    {
      id: 'competitorClassification',
      kicker: 'Classification',
      title: 'Competitive Classification',
      subtitle: 'Position a product using market naming, average price and competitor pressure.',
      shortCode: 'CC',
      buttonLabel: 'Classify product',
    },
    {
      id: 'campaignSuccess',
      kicker: 'Campaigns',
      title: 'Campaign Success',
      subtitle: 'Estimate launch performance from reach, impressions, frequency and budget.',
      shortCode: 'CS',
      buttonLabel: 'Score campaign',
    },
    {
      id: 'customerChurn',
      kicker: 'Retention',
      title: 'Customer Churn Prediction',
      subtitle: 'Score churn probability and review the retention context inside the ML workspace.',
      shortCode: 'CH',
      buttonLabel: 'Score churn risk',
    },
    {
      id: 'bestTimeToSell',
      kicker: 'Forecast',
      title: 'Best Time to Sell',
      subtitle: 'Find the strongest future sales window from historical seasonality.',
      shortCode: 'BS',
      buttonLabel: 'Find best slot',
    },
    {
      id: 'bestTimeToPromote',
      kicker: 'Promotion',
      title: 'Best Time to Promote',
      subtitle: 'Recommend a launch date and discount intensity for your next promotion.',
      shortCode: 'BP',
      buttonLabel: 'Recommend promotion',
    },
    {
      id: 'supplierClassification',
      kicker: 'Suppliers',
      title: 'Supplier Classification',
      subtitle: 'Map a supplier into an operational segment using spend, quantity and cadence.',
      shortCode: 'SC',
      buttonLabel: 'Segment supplier',
    },
  ];

  readonly competitorForm = this.fb.nonNullable.group({
    productName: ['', [Validators.required, Validators.minLength(2)]],
    averagePrice: [45, [Validators.required, Validators.min(0)]],
    competitorCount: [3, [Validators.required, Validators.min(0)]],
  });

  readonly campaignForm = this.fb.nonNullable.group({
    reach: [1200, [Validators.required, Validators.min(0)]],
    impressions: [1800, [Validators.required, Validators.min(0)]],
    frequency: [1.2, [Validators.required, Validators.min(0)]],
    budgetPrice: [3.5, [Validators.required, Validators.min(0)]],
  });

  readonly churnForm = this.fb.nonNullable.group({
    channel: ['All channels', [Validators.required]],
    recency: [90, [Validators.required, Validators.min(0)]],
    frequency: [4, [Validators.required, Validators.min(1)]],
    monetaryTotal: [1200, [Validators.required, Validators.min(0)]],
    monetaryMean: [300, [Validators.required, Validators.min(0)]],
    quantityTotal: [8, [Validators.required, Validators.min(0)]],
    productDiversity: [3, [Validators.required, Validators.min(1)]],
    avgPrice: [150, [Validators.required, Validators.min(0)]],
    customerLifetimeDays: [240, [Validators.required, Validators.min(0)]],
    monetaryStd: [0, [Validators.min(0)]],
  });

  readonly sellForm = this.fb.nonNullable.group({
    targetYear: [2027, [Validators.required, Validators.min(2024)]],
    channel: ['All channels', [Validators.required]],
  });

  readonly promoteForm = this.fb.nonNullable.group({
    targetYear: [2027, [Validators.required, Validators.min(2024)]],
    channel: ['All channels', [Validators.required]],
    objective: ['balanced', [Validators.required]],
  });

  readonly supplierForm = this.fb.nonNullable.group({
    governorate: ['Tunis', [Validators.required, Validators.minLength(2)]],
    city: ['Tunis', [Validators.required, Validators.minLength(2)]],
    totalQuantity: [50, [Validators.required, Validators.min(0)]],
    totalSpend: [3000, [Validators.required, Validators.min(0)]],
    purchaseCount: [6, [Validators.required, Validators.min(0)]],
    averageUnitPrice: [45, [Validators.required, Validators.min(0)]],
    activeProducts: [8, [Validators.required, Validators.min(0)]],
  });

  activeWorkflowId: WorkflowId = 'competitorClassification';
  overview: MlOverviewResponse | null = null;
  churnSummary: CustomerChurnSummaryResponse | null = null;
  loadingOverview = true;
  loadingChurnSummary = true;
  overviewError = '';
  churnSummaryError = '';
  busyWorkflowId: WorkflowId | null = null;

  readonly results: Partial<Record<WorkflowId, MlPredictionResponse>> = {};
  readonly workflowErrors: Partial<Record<WorkflowId, string>> = {};
  readonly submitted: Partial<Record<WorkflowId, boolean>> = {};

  ngOnInit(): void {
    this.loadOverview();
    this.loadChurnSummary();
    this.bindSupplierGeography();
  }

  get activeWorkflow(): WorkflowCard {
    return this.workflows.find((workflow) => workflow.id === this.activeWorkflowId) ?? this.workflows[0];
  }

  get currentForm(): FormGroup {
    return this.getForm(this.activeWorkflowId);
  }

  get channels(): string[] {
    return this.overview?.options.channels ?? ['All channels'];
  }

  get targetYears(): number[] {
    return this.overview?.options.targetYears ?? [2026, 2027];
  }

  get promotionObjectives(): Array<{ value: string; label: string }> {
    return (
      this.overview?.options.promotionObjectives ?? [
        { value: 'balanced', label: 'Balanced' },
        { value: 'volume', label: 'Volume' },
        { value: 'margin', label: 'Margin' },
      ]
    );
  }

  get governorates(): string[] {
    return this.overview?.options.governorates ?? ['Tunis'];
  }

  get citiesForGovernorate(): string[] {
    const governorate = this.supplierForm.controls.governorate.value;
    return this.overview?.options.citiesByGovernorate?.[governorate] ?? ['Tunis'];
  }

  get topRiskCustomers(): CustomerChurnCustomerRow[] {
    return this.churnSummary?.customers.slice(0, 5) ?? [];
  }

  setActiveWorkflow(workflowId: WorkflowId): void {
    this.activeWorkflowId = workflowId;
  }

  submitActiveWorkflow(): void {
    const workflowId = this.activeWorkflowId;
    const form = this.getForm(workflowId);
    this.submitted[workflowId] = true;
    this.workflowErrors[workflowId] = '';

    if (form.invalid) {
      form.markAllAsTouched();
      return;
    }

    this.busyWorkflowId = workflowId;
    const request = this.resolveRequest(workflowId, form.getRawValue());
    request.subscribe({
      next: (response) => {
        this.results[workflowId] = response;
        this.busyWorkflowId = null;
      },
      error: (error) => {
        this.workflowErrors[workflowId] = this.mlService.describeError(error).message;
        this.busyWorkflowId = null;
      },
    });
  }

  isBusy(workflowId: WorkflowId): boolean {
    return this.busyWorkflowId === workflowId;
  }

  resultFor(workflowId: WorkflowId): MlPredictionResponse | null {
    return this.results[workflowId] ?? null;
  }

  confidenceWidth(result: MlPredictionResponse | null): string {
    if (!result) {
      return '0%';
    }
    return `${Math.max(8, Math.round(result.confidence * 100))}%`;
  }

  riskShareWidth(share: number): string {
    return `${Math.max(10, Math.round(share * 100))}%`;
  }

  trackByCustomer(_: number, customer: CustomerChurnCustomerRow): number {
    return customer.customerId;
  }

  showInvalid(form: FormGroup, controlName: string): boolean {
    const control = form.get(controlName);
    return !!control && control.invalid && (control.touched || Boolean(this.submitted[this.activeWorkflowId]));
  }

  controlMessage(form: FormGroup, controlName: string, label: string): string {
    const control = form.get(controlName);
    if (!control?.errors) {
      return '';
    }

    if (control.errors['required']) {
      return `${label} est requis.`;
    }

    if (control.errors['minlength']) {
      return `${label} doit contenir plus de caracteres.`;
    }

    if (control.errors['min']) {
      return `${label} doit etre positif.`;
    }

    return `${label} est invalide.`;
  }

  resultHighlights(result: MlPredictionResponse): Array<{ label: string; value: string }> {
    switch (result.workflow) {
      case 'competitorClassification':
        return [
          { label: 'Predicted category', value: String(result.result['predictedCategory'] ?? '-') },
          { label: 'Recommendation', value: String(result.result['recommendationLevel'] ?? '-') },
          { label: 'Market price', value: `${result.result['averageCategoryPrice'] ?? '-'} TND` },
        ];
      case 'campaignSuccess':
        return [
          { label: 'Probability', value: `${Math.round(Number(result.result['successProbability'] ?? 0) * 100)}%` },
          { label: 'Projected views', value: String(result.result['predictedViews'] ?? '-') },
          { label: 'Predicted result', value: String(result.result['predictedResult'] ?? '-') },
        ];
      case 'customerChurn':
        return [
          { label: 'Risk band', value: String(result.result['riskBand'] ?? '-') },
          { label: 'Segment', value: String(result.result['segmentName'] ?? '-') },
          { label: 'Action', value: String(result.result['recommendedAction'] ?? '-') },
        ];
      case 'bestTimeToSell':
        return [
          { label: 'Best date', value: String(result.result['bestDate'] ?? '-') },
          { label: 'Best month', value: String(result.result['bestMonth'] ?? '-') },
          { label: 'Best weekday', value: String(result.result['bestWeekday'] ?? '-') },
        ];
      case 'bestTimeToPromote':
        return [
          { label: 'Launch date', value: String(result.result['launchDate'] ?? '-') },
          { label: 'Discount', value: `${result.result['recommendedDiscountPercent'] ?? '-'}%` },
          { label: 'Expected uplift', value: `${result.result['expectedUpliftPercent'] ?? '-'}%` },
        ];
      case 'supplierClassification':
        return [
          { label: 'Profile', value: String(result.result['profileName'] ?? '-') },
          { label: 'Cluster', value: String(result.result['clusterId'] ?? '-') },
          { label: 'Action', value: String(result.result['nextAction'] ?? '-') },
        ];
      default:
        return [];
    }
  }

  supplementaryCards(result: MlPredictionResponse): Array<{ title: string; value: string; caption: string }> {
    switch (result.workflow) {
      case 'competitorClassification':
        return (
          (result.result['relatedProducts'] as Array<Record<string, unknown>> | undefined)?.map((item) => ({
            title: String(item['product'] ?? '-'),
            value: `${item['averagePrice'] ?? '-'} TND`,
            caption: `${item['competitorCount'] ?? '-'} competitors`,
          })) ?? []
        );
      case 'bestTimeToSell':
        return (
          (result.result['topWindows'] as Array<Record<string, unknown>> | undefined)?.map((item) => ({
            title: String(item['date'] ?? '-'),
            value: `${item['predictedRevenue'] ?? '-'} TND`,
            caption: 'forecasted revenue',
          })) ?? []
        );
      case 'bestTimeToPromote':
        return (
          (result.result['topWindows'] as Array<Record<string, unknown>> | undefined)?.map((item) => ({
            title: `${item['month'] ?? '-'}`,
            value: `${item['score'] ?? '-'} score`,
            caption: String(item['weekday'] ?? ''),
          })) ?? []
        );
      case 'supplierClassification':
        return (
          (result.result['peerSuppliers'] as Array<Record<string, unknown>> | undefined)?.map((item) => ({
            title: String(item['supplier'] ?? '-'),
            value: `${item['totalSpend'] ?? '-'} TND`,
            caption: `${item['city'] ?? '-'}, ${item['governorate'] ?? '-'}`,
          })) ?? []
        );
      default:
        return [];
    }
  }

  private loadOverview(): void {
    this.loadingOverview = true;
    this.overviewError = '';
    this.mlService.getOverview().subscribe({
      next: (overview) => {
        this.overview = overview;
        this.loadingOverview = false;
        this.patchDynamicDefaults(overview);
      },
      error: (error) => {
        this.overviewError = this.mlService.describeError(error).message;
        this.loadingOverview = false;
      },
    });
  }

  private loadChurnSummary(): void {
    this.loadingChurnSummary = true;
    this.churnSummaryError = '';
    this.mlService.getCustomerChurnSummary().subscribe({
      next: (summary) => {
        this.churnSummary = summary;
        this.loadingChurnSummary = false;
        this.patchChurnDefaults(summary);
      },
      error: (error) => {
        this.churnSummaryError = this.mlService.describeError(error).message;
        this.loadingChurnSummary = false;
      },
    });
  }

  private bindSupplierGeography(): void {
    this.supplierForm.controls.governorate.valueChanges
      .pipe(startWith(this.supplierForm.controls.governorate.value), takeUntilDestroyed(this.destroyRef))
      .subscribe((governorate) => {
        const nextCities = this.overview?.options.citiesByGovernorate?.[governorate] ?? [];
        if (nextCities.length === 0) {
          return;
        }

        if (!nextCities.includes(this.supplierForm.controls.city.value)) {
          this.supplierForm.controls.city.setValue(nextCities[0]);
        }
      });
  }

  private patchDynamicDefaults(overview: MlOverviewResponse): void {
    const defaultYear = overview.options.targetYears.at(-1) ?? this.sellForm.controls.targetYear.value;
    const defaultChannel = overview.options.channels[0] ?? 'All channels';
    const defaultGovernorate = overview.options.governorates[0] ?? 'Tunis';
    const defaultCity = overview.options.citiesByGovernorate[defaultGovernorate]?.[0] ?? 'Tunis';

    this.sellForm.patchValue({ targetYear: defaultYear, channel: defaultChannel });
    this.promoteForm.patchValue({ targetYear: defaultYear, channel: defaultChannel });
    this.supplierForm.patchValue({ governorate: defaultGovernorate, city: defaultCity });
    this.churnForm.patchValue({ channel: defaultChannel });
  }

  private patchChurnDefaults(summary: CustomerChurnSummaryResponse): void {
    const defaultChannel = summary.channelSummary[0]?.channel ?? this.churnForm.controls.channel.value;
    const mediumRiskCustomer =
      summary.customers.find((customer) => customer.riskBand === 'Medium') ?? summary.customers[0];

    this.churnForm.patchValue({
      channel: defaultChannel,
      recency: mediumRiskCustomer?.recency ?? 90,
      frequency: mediumRiskCustomer?.frequency ?? 4,
      monetaryTotal: mediumRiskCustomer?.monetaryTotal ?? 1200,
      monetaryMean:
        mediumRiskCustomer && mediumRiskCustomer.frequency > 0
          ? Math.round((mediumRiskCustomer.monetaryTotal / mediumRiskCustomer.frequency) * 100) / 100
          : 300,
      quantityTotal: mediumRiskCustomer?.frequency ?? 8,
      productDiversity: Math.max(2, Math.round((mediumRiskCustomer?.rfmScore ?? 6) / 2)),
      avgPrice:
        mediumRiskCustomer && mediumRiskCustomer.frequency > 0
          ? Math.round((mediumRiskCustomer.monetaryTotal / mediumRiskCustomer.frequency) * 100) / 100
          : 150,
      customerLifetimeDays: Math.max(120, (mediumRiskCustomer?.recency ?? 90) * 2),
      monetaryStd: 0,
    });
  }

  private getForm(workflowId: WorkflowId): FormGroup {
    switch (workflowId) {
      case 'competitorClassification':
        return this.competitorForm;
      case 'campaignSuccess':
        return this.campaignForm;
      case 'customerChurn':
        return this.churnForm;
      case 'bestTimeToSell':
        return this.sellForm;
      case 'bestTimeToPromote':
        return this.promoteForm;
      case 'supplierClassification':
        return this.supplierForm;
    }
  }

  private resolveRequest(workflowId: WorkflowId, payload: Record<string, unknown>) {
    switch (workflowId) {
      case 'competitorClassification':
        return this.mlService.classifyCompetitor(payload as never);
      case 'campaignSuccess':
        return this.mlService.predictCampaignSuccess(payload as never);
      case 'customerChurn':
        return this.mlService.predictCustomerChurn(payload as never);
      case 'bestTimeToSell':
        return this.mlService.predictBestTimeToSell(payload as never);
      case 'bestTimeToPromote':
        return this.mlService.predictBestTimeToPromote(payload as never);
      case 'supplierClassification':
        return this.mlService.classifySupplier(payload as never);
    }
  }
}
