import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { Observable, TimeoutError, retry, timeout } from 'rxjs';

export interface MlOverviewMetric {
  label: string;
  value: number | string;
  hint: string;
}

export interface MlPredictionMetric {
  label: string;
  value: string;
  tone: 'primary' | 'success' | 'neutral';
}

export interface MlPredictionResponse {
  workflow: string;
  headline: string;
  summary: string;
  status: 'strong' | 'balanced' | 'watch' | string;
  confidence: number;
  metrics: MlPredictionMetric[];
  insights: string[];
  result: Record<string, unknown>;
}

export interface MlOverviewResponse {
  status: string;
  databaseStatus: {
    enabled: boolean;
    connected: boolean;
    message: string;
    schema: string;
  };
  datasets: Record<string, string>;
  metrics: MlOverviewMetric[];
  options: {
    channels: string[];
    targetYears: number[];
    governorates: string[];
    citiesByGovernorate: Record<string, string[]>;
    souguiCategories: string[];
    promotionObjectives: Array<{ value: string; label: string }>;
  };
  workflows: Array<{
    id: string;
    title: string;
    subtitle: string;
  }>;
}

export interface CustomerChurnSummaryMetric {
  label: string;
  value: number | string;
  hint: string;
}

export interface CustomerChurnPerformance {
  best_model: string;
  roc_auc: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  evaluated_customers: number;
  feature_count: number;
}

export interface CustomerChurnRiskBand {
  risk_band: string;
  customers: number;
  share: number;
}

export interface CustomerChurnChannelSummary {
  channel: string;
  totalCustomers: number;
  churnRate: number;
  avgRecency: number;
  avgFrequency: number;
  avgChurnProbability: number;
}

export interface CustomerChurnSegmentSummary {
  segment: number;
  segmentName: string;
  customers: number;
  churnRate: number;
  avgChurnProbability: number;
  avgRecency: number;
  avgFrequency: number;
  avgMonetary: number;
  avgRfmScore: number;
}

export interface CustomerChurnFeatureImportance {
  feature: string;
  weight: number;
}

export interface CustomerChurnCustomerRow {
  customerId: number;
  customerName: string;
  customerCode: string;
  channel: string;
  segment: string;
  riskBand: string;
  recency: number;
  frequency: number;
  monetaryTotal: number;
  rfmScore: number;
  churnProbability: number;
  recommendedAction: string;
}

export interface CustomerChurnSummaryResponse {
  workflow: string;
  headline: string;
  summary: string;
  referenceDate: string;
  churnThresholdDays: number;
  metrics: CustomerChurnSummaryMetric[];
  modelPerformance: CustomerChurnPerformance;
  insights: string[];
  riskDistribution: CustomerChurnRiskBand[];
  channelSummary: CustomerChurnChannelSummary[];
  segmentSummary: CustomerChurnSegmentSummary[];
  featureImportance: CustomerChurnFeatureImportance[];
  customers: CustomerChurnCustomerRow[];
}

export interface CustomerChurnScenarioPayload {
  channel: string;
  recency: number;
  frequency: number;
  monetaryTotal: number;
  monetaryMean: number;
  quantityTotal: number;
  productDiversity: number;
  avgPrice: number;
  customerLifetimeDays: number;
  monetaryStd?: number;
}

export interface ProductMatcherProduct {
  productId: number;
  imageName: string;
  productName: string;
  mainCategory: string;
  subcategory: string;
  productUrl: string;
  imageUrl: string;
  similarityScore?: number;
  visualScore?: number;
  textScore?: number;
}

export interface ProductMatcherCategoryMetric {
  name: string;
  products: number;
  share: number;
}

export interface ProductMatcherOverviewResponse {
  workflow: string;
  headline: string;
  summary: string;
  embeddingBackend: string;
  metrics: MlOverviewMetric[];
  categories: ProductMatcherCategoryMetric[];
  featuredProducts: ProductMatcherProduct[];
}

export interface ProductMatcherCatalogResponse {
  query: string;
  category: string;
  total: number;
  products: ProductMatcherProduct[];
}

export interface ProductMatcherPredictedCategory {
  name: string;
  matches: number;
}

export interface ProductMatcherCompareResponse {
  workflow: string;
  headline: string;
  summary: string;
  embeddingBackend: string;
  queryProduct: ProductMatcherProduct;
  metrics: MlPredictionMetric[];
  predictedCategories: ProductMatcherPredictedCategory[];
  matches: ProductMatcherProduct[];
}

export interface MlErrorDetails {
  kind: 'timeout' | 'offline' | 'server' | 'unknown';
  message: string;
  status?: number;
}

@Injectable({ providedIn: 'root' })
export class MlService {
  private readonly overviewEndpoint = '/api/ml/overview';
  private readonly competitorEndpoint = '/api/ml/competitors/classify';
  private readonly campaignEndpoint = '/api/ml/campaigns/predict-success';
  private readonly sellEndpoint = '/api/ml/sales/best-time';
  private readonly promoteEndpoint = '/api/ml/promotions/best-time';
  private readonly supplierEndpoint = '/api/ml/suppliers/classify';
  private readonly churnSummaryEndpoint = '/api/ml/customers/churn/summary';
  private readonly churnPredictEndpoint = '/api/ml/customers/churn/predict';
  private readonly matcherOverviewEndpoint = '/api/ml/matcher/overview';
  private readonly matcherCatalogEndpoint = '/api/ml/matcher/catalog';
  private readonly matcherCompareEndpoint = '/api/ml/matcher/compare';

  constructor(private readonly http: HttpClient) {}

  getOverview(): Observable<MlOverviewResponse> {
    return this.http.get<MlOverviewResponse>(this.overviewEndpoint).pipe(timeout(12000));
  }

  classifyCompetitor(payload: {
    productName: string;
    averagePrice: number;
    competitorCount: number;
  }): Observable<MlPredictionResponse> {
    return this.postPrediction(this.competitorEndpoint, payload);
  }

  predictCampaignSuccess(payload: {
    reach: number;
    impressions: number;
    frequency: number;
    budgetPrice: number;
  }): Observable<MlPredictionResponse> {
    return this.postPrediction(this.campaignEndpoint, payload);
  }

  predictBestTimeToSell(payload: {
    targetYear: number;
    channel: string;
  }): Observable<MlPredictionResponse> {
    return this.postPrediction(this.sellEndpoint, payload);
  }

  predictBestTimeToPromote(payload: {
    targetYear: number;
    channel: string;
    objective: string;
  }): Observable<MlPredictionResponse> {
    return this.postPrediction(this.promoteEndpoint, payload);
  }

  classifySupplier(payload: {
    governorate: string;
    city: string;
    totalQuantity: number;
    totalSpend: number;
    purchaseCount: number;
    averageUnitPrice: number;
    activeProducts: number;
  }): Observable<MlPredictionResponse> {
    return this.postPrediction(this.supplierEndpoint, payload);
  }

  getCustomerChurnSummary(): Observable<CustomerChurnSummaryResponse> {
    return this.http.get<CustomerChurnSummaryResponse>(this.churnSummaryEndpoint).pipe(timeout(20000));
  }

  predictCustomerChurn(payload: CustomerChurnScenarioPayload): Observable<MlPredictionResponse> {
    return this.postPrediction(this.churnPredictEndpoint, payload);
  }

  getMatcherOverview(): Observable<ProductMatcherOverviewResponse> {
    return this.http.get<ProductMatcherOverviewResponse>(this.matcherOverviewEndpoint).pipe(timeout(20000));
  }

  searchMatcherCatalog(params: {
    query?: string;
    category?: string;
    limit?: number;
  }): Observable<ProductMatcherCatalogResponse> {
    let httpParams = new HttpParams();

    if (params.query) {
      httpParams = httpParams.set('query', params.query);
    }

    if (params.category) {
      httpParams = httpParams.set('category', params.category);
    }

    if (typeof params.limit === 'number') {
      httpParams = httpParams.set('limit', String(params.limit));
    }

    return this.http
      .get<ProductMatcherCatalogResponse>(this.matcherCatalogEndpoint, { params: httpParams })
      .pipe(timeout(20000));
  }

  compareMatcherProducts(payload: {
    imageName?: string;
    productName?: string;
    uploadedImageDataUrl?: string;
    uploadedImageName?: string;
    limit?: number;
  }): Observable<ProductMatcherCompareResponse> {
    return this.http.post<ProductMatcherCompareResponse>(this.matcherCompareEndpoint, payload).pipe(
      timeout(30000),
      retry({ count: 1, delay: 700 })
    );
  }

  describeError(error: unknown): MlErrorDetails {
    if (error instanceof TimeoutError) {
      return {
        kind: 'timeout',
        message: 'Le service ML met trop de temps a repondre. Relancez la prediction.'
      };
    }

    if (error instanceof HttpErrorResponse) {
      if (error.status === 0) {
        return {
          kind: 'offline',
          message: "L'API ML est inaccessible. Lancez `npm start` ou `npm run start:api`, puis reessayez.",
          status: 0
        };
      }

      const detail =
        typeof error.error?.detail === 'string'
          ? error.error.detail
          : 'Le service ML a retourne une erreur inattendue.';

      return {
        kind: 'server',
        message: detail,
        status: error.status
      };
    }

    return {
      kind: 'unknown',
      message: 'La prediction n a pas pu etre terminee.'
    };
  }

  private postPrediction<T extends MlPredictionResponse>(
    endpoint: string,
    payload: object
  ): Observable<T> {
    return this.http.post<T>(endpoint, payload).pipe(
      timeout(30000),
      retry({ count: 1, delay: 700 })
    );
  }
}
