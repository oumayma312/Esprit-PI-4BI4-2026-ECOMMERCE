import { Injectable } from '@angular/core';
import { HttpClient, HttpEventType } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { timeout, catchError, tap } from 'rxjs/operators';

const API = '/api';

function logApi(method: string, url: string, body?: any) {
  console.log(`[API] ${method} ${url}`, body ? { body } : {});
}

function logApiResponse(url: string, status: number, data?: any) {
  console.log(`[API] ${url} → ${status}`, data && typeof data === 'object' && Object.keys(data).length < 10 ? data : '');
}

function logApiError(url: string, error: any) {
  console.error(`[API] ${url} FAILED`, { status: error.status, message: error.message, error });
}

export interface Manufacturer {
  id?: number;
  name: string;
  email: string;
  phone: string;
  location: string;
  materials_supplied: string;
}

export interface Material {
  id?: number;
  name: string;
  category: string;
  unit_price: number;
  unit: string;
  min_order_qty: number;
  stock_quantity?: number;
  manufacturer_id?: number;
  manufacturer_name?: string;
}

export interface Order {
  id?: number;
  material_id: number;
  quantity: number;
  status: string;
  notes: string;
  created_at?: string;
  material_name?: string;
  category?: string;
  manufacturer_name?: string;
  manufacturer_email?: string;
}

export interface ChatResponse {
  thread_id: string;
  response: string;
  pending_approval: boolean;
  tool_calls: any[];
  interrupt_data?: any;
}

export interface InterruptData {
  name: string;
  arguments: any;
  description: string;
}

export interface PrefillData {
  materials: Array<{ material_id: number | null; quantity: number }>;
  notes?: string;
}

export interface EditDecision {
  thread_id: string;
  tool_name: string;
  edited_args: any;
}

export interface CustomerPrediction {
  id: number;
  customer_id: number;
  churn_probability: number;
  churn_prediction: number | string;
  segment: number;
  segment_label: string;
  risk_level: string;
  recommendation: string;
  predicted_at?: string;
}

export interface Customer {
  id: number;
  name: string;
  code?: string;
  channel_id?: number;
  active?: number;
  churn_probability?: number;
  churn_prediction?: string | number;
  segment?: number;
  segment_label?: string;
  risk_level?: string;
  recommendation?: string;
  predicted_at?: string;
  prediction?: CustomerPrediction;
  transactions?: any[];
}

export interface CustomerFeatures {
  recency: number;
  frequency: number;
  monetary_total: number;
  monetary_trend: number;
  product_diversity: number;
  channel_diversity: number;
  lifetime_days: number;
  purchase_velocity: number;
  avg_price: number;
  channel?: string;
}

export interface SegmentStat {
  segment: number;
  segment_label: string;
  count: number;
  avg_probability: number;
  min_probability: number;
  max_probability: number;
}

export interface CustomerFilters {
  segment?: number | null;
  risk_level?: string | null;
  search?: string;
  limit?: number;
  offset?: number;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  // Manufacturers
  getManufacturers(): Observable<Manufacturer[]> {
    return this.http.get<Manufacturer[]>(`${API}/manufacturers`);
  }

  createManufacturer(m: Manufacturer): Observable<Manufacturer> {
    return this.http.post<Manufacturer>(`${API}/manufacturers`, m);
  }

  updateManufacturer(id: number, m: Partial<Manufacturer>): Observable<Manufacturer> {
    return this.http.put<Manufacturer>(`${API}/manufacturers/${id}`, m);
  }

  deleteManufacturer(id: number): Observable<void> {
    return this.http.delete<void>(`${API}/manufacturers/${id}`, { responseType: 'text' as any });
  }

  // Materials
  getMaterials(): Observable<Material[]> {
    return this.http.get<Material[]>(`${API}/materials`);
  }

  createMaterial(m: Material): Observable<Material> {
    return this.http.post<Material>(`${API}/materials`, m);
  }

  updateMaterial(id: number, m: Partial<Material>): Observable<Material> {
    return this.http.put<Material>(`${API}/materials/${id}`, m);
  }

  deleteMaterial(id: number): Observable<void> {
    return this.http.delete<void>(`${API}/materials/${id}`, { responseType: 'text' as any });
  }

  // Materials context for edit panel
  getMaterialsContext(category?: string): Observable<Material[]> {
    if (category) {
      return this.http.get<Material[]>(`${API}/materials/context?category=${encodeURIComponent(category)}`);
    }
    return this.http.get<Material[]>(`${API}/materials/context`);
  }

  // Orders
  getOrders(): Observable<Order[]> {
    return this.http.get<Order[]>(`${API}/orders`);
  }

  createOrder(o: Order): Observable<Order> {
    return this.http.post<Order>(`${API}/orders`, o);
  }

  updateOrderStatus(id: number, status: string): Observable<Order> {
    return this.http.patch<Order>(`${API}/orders/${id}/status`, { status });
  }

  deleteOrder(id: number): Observable<void> {
    return this.http.delete<void>(`${API}/orders/${id}`, { responseType: 'text' as any });
  }

  // Chat
  chat(message: string, threadId?: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${API}/chat`, { message, thread_id: threadId });
  }

  // SSE streaming chat
  streamChat(message: string, threadId?: string): EventSource {
    const url = `${API}/chat/stream?message=${encodeURIComponent(message)}${threadId ? `&thread_id=${threadId}` : ''}`;
    return new EventSource(url);
  }

  approveChat(threadId: string): Observable<any> {
    return this.http.post(`${API}/chat/approve`, { thread_id: threadId, decision: 'approve' });
  }

  rejectChat(threadId: string): Observable<any> {
    return this.http.post(`${API}/chat/reject`, { thread_id: threadId, decision: 'reject' });
  }

  editChat(threadId: string, toolName: string, editedArgs: any): Observable<any> {
    return this.http.post(`${API}/chat/edit`, {
      thread_id: threadId,
      tool_name: toolName,
      edited_args: editedArgs
    });
  }

  // --- ML Prediction ---
  predictChurn(features: CustomerFeatures): Observable<any> {
    logApi('POST', `${API}/predict/churn`, features);
    return this.http.post(`${API}/predict/churn`, features).pipe(
      timeout(15000),
      tap(
        (data) => {
          logApiResponse(`${API}/predict/churn`, 200, data);
        },
        (error) => {
          logApiError(`${API}/predict/churn`, error);
        }
      ),
      catchError((error) => {
        return throwError(() => error);
      })
    );
  }

  predictChurnBatch(data: any[]): Observable<any> {
    logApi('POST', `${API}/predict/churn/batch`, { count: data.length });
    return this.http.post(`${API}/predict/churn/batch`, { data }).pipe(
      catchError((error) => {
        logApiError(`${API}/predict/churn/batch`, error);
        return throwError(() => error);
      })
    );
  }

  // --- Customers ---
  listCustomers(params?: CustomerFilters): Observable<Customer[]> {
    const query = new URLSearchParams();
    if (params?.segment !== null && params?.segment !== undefined) query.set('segment', String(params.segment));
    if (params?.risk_level) query.set('risk_level', params.risk_level);
    if (params?.search) query.set('search', params.search);
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset !== undefined && params?.offset !== null) query.set('offset', String(params.offset));
    const qs = query.toString();
    return this.http.get<Customer[]>(`${API}/customers${qs ? '?' + qs : ''}`);
  }

  getCustomerCount(params?: { segment?: number | null; risk_level?: string | null; search?: string }): Observable<{ count: number }> {
    const query = new URLSearchParams();
    if (params?.segment !== null && params?.segment !== undefined) query.set('segment', String(params.segment));
    if (params?.risk_level) query.set('risk_level', params.risk_level);
    if (params?.search) query.set('search', params.search);
    const qs = query.toString();
    return this.http.get<{ count: number }>(`${API}/customers/count${qs ? '?' + qs : ''}`);
  }

  getCustomer(id: number): Observable<Customer> {
    return this.http.get<Customer>(`${API}/customers/${id}`);
  }

  getHighRiskCustomers(minProbability?: number): Observable<Customer[]> {
    const qs = minProbability !== undefined ? `?min_probability=${minProbability}` : '';
    return this.http.get<Customer[]>(`${API}/customers/high-risk${qs}`);
  }

  getSegmentStats(): Observable<SegmentStat[]> {
    return this.http.get<SegmentStat[]>(`${API}/customers/segments`);
  }

  getPCAPlot(): Observable<any> {
    return this.http.get(`${API}/predict/pca-plot`, { responseType: 'blob' as any });
  }
}
