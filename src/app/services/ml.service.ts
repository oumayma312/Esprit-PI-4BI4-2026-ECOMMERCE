import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, TimeoutError, retry, timeout } from 'rxjs';

export interface MlOverviewMetric {
  label: string;
  value: number;
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
    promotionObjectives: Array<{ value: string; label: string }>;
  };
  workflows: Array<{
    id: string;
    title: string;
    subtitle: string;
  }>;
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
          : 'Le backend ML a retourne une erreur inattendue.';

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
    payload: Record<string, unknown>
  ): Observable<T> {
    return this.http.post<T>(endpoint, payload).pipe(
      timeout(30000),
      retry({ count: 1, delay: 700 })
    );
  }
}

