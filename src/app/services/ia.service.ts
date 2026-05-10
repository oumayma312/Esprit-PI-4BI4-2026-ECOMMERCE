import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PipelineStatus {
  status: string;
  modules: string[];
}

export interface PipelineRunResult {
  status: string;
  result: Record<string, unknown>;
}

export interface SchedulerStatus {
  status: string;
  interval: string;
  next_run_utc: string;
  next_run_in_seconds: number;
}

export interface DecisionsSummary {
  total: number;
  by_role: Record<string, number>;
  unread: number;
}

export interface DecisionItem {
  id: number;
  date_generated: string;
  role: string;
  titre: string;
  urgence: string;
  sources: string[];
  statut: string;
  created_at: string;
}

export interface MlModelArtifact {
  model_key: string;
  kind?: string;
  description?: string;
}

export interface MlModelsList {
  models: MlModelArtifact[];
}

@Injectable({ providedIn: 'root' })
export class IaService {
  private readonly pipelineRunEndpoint = '/api/pipeline/run';
  private readonly pipelineStatusEndpoint = '/api/pipeline/status';
  private readonly schedulerStatusEndpoint = '/api/scheduler/status';
  private readonly decisionsEndpoint = '/api/decisions';
  private readonly decisionsSummaryEndpoint = '/api/decisions/summary';
  private readonly mlModelsEndpoint = '/api/ml/models';

  constructor(private readonly http: HttpClient) {}

  getPipelineStatus(): Observable<PipelineStatus> {
    return this.http.get<PipelineStatus>(this.pipelineStatusEndpoint);
  }

  runPipeline(): Observable<PipelineRunResult> {
    return this.http.post<PipelineRunResult>(this.pipelineRunEndpoint, {});
  }

  getSchedulerStatus(): Observable<SchedulerStatus> {
    return this.http.get<SchedulerStatus>(this.schedulerStatusEndpoint);
  }

  getDecisionsSummary(): Observable<DecisionsSummary> {
    return this.http.get<DecisionsSummary>(this.decisionsSummaryEndpoint);
  }

  getDecisions(
    role: string,
    options?: { unreadOnly?: boolean; limit?: number; includeSnapshot?: boolean },
  ): Observable<DecisionItem[]> {
    let params = new HttpParams().set('role', role).set('limit', String(options?.limit ?? 50));

    if (options?.unreadOnly) {
      params = params.set('unread_only', 'true');
    }

    if (options?.includeSnapshot) {
      params = params.set('include_snapshot', 'true');
    }

    return this.http.get<DecisionItem[]>(this.decisionsEndpoint, { params });
  }

  markDecisionRead(decisionId: number): Observable<{ decision_id: number; statut: string }> {
    return this.http.post<{ decision_id: number; statut: string }>(
      `${this.decisionsEndpoint}/${decisionId}/read`,
      {},
    );
  }

  getMlModels(): Observable<MlModelsList> {
    return this.http.get<MlModelsList>(this.mlModelsEndpoint);
  }
}
