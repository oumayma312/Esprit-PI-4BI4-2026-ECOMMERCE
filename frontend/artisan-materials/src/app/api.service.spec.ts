import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { HttpClient, HttpHandler } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ApiService, Material, Order, Manufacturer } from './api.service';

// Mock EventSource for jsdom
class MockEventSource {
  url: string;
  constructor(url: string) {
    this.url = url;
  }
  close() {}
}
globalThis.EventSource = MockEventSource as any;

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: {
    get: ReturnType<typeof vi.fn>;
    post: ReturnType<typeof vi.fn>;
    put: ReturnType<typeof vi.fn>;
    patch: ReturnType<typeof vi.fn>;
    delete: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    const getSpy = vi.fn().mockReturnValue(new Observable(() => {}));
    const postSpy = vi.fn().mockReturnValue(new Observable(() => {}));
    const putSpy = vi.fn().mockReturnValue(new Observable(() => {}));
    const patchSpy = vi.fn().mockReturnValue(new Observable(() => {}));
    const deleteSpy = vi.fn().mockReturnValue(new Observable(() => {}));

    TestBed.configureTestingModule({
      providers: [
        ApiService,
        {
          provide: HttpClient,
          useValue: { get: getSpy, post: postSpy, put: putSpy, patch: patchSpy, delete: deleteSpy }
        }
      ]
    });
    service = TestBed.inject(ApiService);
    httpMock = {
      get: getSpy,
      post: postSpy,
      put: putSpy,
      patch: patchSpy,
      delete: deleteSpy,
    };
  });

  describe('URL construction', () => {
    it('should construct correct URL for getMaterials', () => {
      service.getMaterials().subscribe();
      expect(httpMock.get).toHaveBeenCalledWith('/api/materials');
    });

    it('should construct correct URL for createMaterial', () => {
      const mat: Material = { name: 'Test', category: 'Test', unit_price: 1.0, unit: 'piece', min_order_qty: 1 };
      service.createMaterial(mat).subscribe();
      expect(httpMock.post).toHaveBeenCalledWith('/api/materials', mat);
    });

    it('should construct correct URL for updateMaterial', () => {
      service.updateMaterial(1, { name: 'Updated' }).subscribe();
      expect(httpMock.put).toHaveBeenCalledWith('/api/materials/1', { name: 'Updated' });
    });

    it('should construct correct URL for deleteMaterial', () => {
      service.deleteMaterial(1).subscribe();
      expect(httpMock.delete).toHaveBeenCalledOnce();
      expect(httpMock.delete.mock.calls[0][0]).toBe('/api/materials/1');
    });

    it('should construct correct URL for getMaterialsContext', () => {
      service.getMaterialsContext('Metal').subscribe();
      expect(httpMock.get).toHaveBeenCalledWith('/api/materials/context?category=Metal');
    });

    it('should construct correct URL for getMaterialsContext without category', () => {
      service.getMaterialsContext().subscribe();
      expect(httpMock.get).toHaveBeenCalledWith('/api/materials/context');
    });

    it('should construct correct URL for createOrder', () => {
      const order: Order = { material_id: 1, quantity: 10, status: 'pending', notes: '' };
      service.createOrder(order).subscribe();
      expect(httpMock.post).toHaveBeenCalledWith('/api/orders', order);
    });

    it('should construct correct URL for updateOrderStatus', () => {
      service.updateOrderStatus(1, 'approved').subscribe();
      expect(httpMock.patch).toHaveBeenCalledWith('/api/orders/1/status', { status: 'approved' });
    });

    it('should construct correct URL for getOrders', () => {
      service.getOrders().subscribe();
      expect(httpMock.get).toHaveBeenCalledWith('/api/orders');
    });

    it('should construct correct URL for chat', () => {
      service.chat('Hello').subscribe();
      expect(httpMock.post).toHaveBeenCalledWith('/api/chat', { message: 'Hello', thread_id: undefined });
    });

    it('should construct correct URL for chat with thread_id', () => {
      service.chat('Hello', 'thread-123').subscribe();
      expect(httpMock.post).toHaveBeenCalledWith('/api/chat', { message: 'Hello', thread_id: 'thread-123' });
    });

    it('should construct correct URL for approveChat', () => {
      service.approveChat('thread-123').subscribe();
      expect(httpMock.post).toHaveBeenCalledWith('/api/chat/approve', {
        thread_id: 'thread-123',
        decision: 'approve'
      });
    });

    it('should construct correct URL for rejectChat', () => {
      service.rejectChat('thread-123').subscribe();
      expect(httpMock.post).toHaveBeenCalledWith('/api/chat/reject', {
        thread_id: 'thread-123',
        decision: 'reject'
      });
    });

    it('should construct correct URL for editChat', () => {
      service.editChat('thread-123', 'create_order', { material_id: 1, quantity: 50 }).subscribe();
      expect(httpMock.post).toHaveBeenCalledWith('/api/chat/edit', {
        thread_id: 'thread-123',
        tool_name: 'create_order',
        edited_args: { material_id: 1, quantity: 50 }
      });
    });

    it('should construct correct URL for streamChat', () => {
      const es = service.streamChat('Hello');
      expect(es).toBeInstanceOf(MockEventSource);
    });

    it('should encode URL parameters in streamChat', () => {
      const es = service.streamChat('I need 50 meters of copper wire');
      expect(es.url).toContain(encodeURIComponent('I need 50 meters of copper wire'));
    });

    it('should include thread_id in streamChat URL', () => {
      const es = service.streamChat('Hello', 'thread-456');
      expect(es.url).toContain('thread_id=thread-456');
    });
  });

  describe('Material interface', () => {
    it('should support stock_quantity property', () => {
      const mat: Material = {
        id: 1, name: 'Copper Wire', category: 'Metal',
        unit_price: 4.50, unit: 'meter', min_order_qty: 10,
        stock_quantity: 150
      };
      expect(mat.stock_quantity).toBe(150);
    });

    it('should support manufacturer_name property', () => {
      const mat: Material = {
        id: 1, name: 'Copper Wire', category: 'Metal',
        unit_price: 4.50, unit: 'meter', min_order_qty: 10,
        manufacturer_name: 'Test Mfr'
      };
      expect(mat.manufacturer_name).toBe('Test Mfr');
    });
  });

  describe('Manufacturer interface', () => {
    it('should have all required properties', () => {
      const mfr: Manufacturer = {
        id: 1, name: 'Test Mfr', email: 'test@test.com',
        phone: '+123', location: 'Here', materials_supplied: 'Copper'
      };
      expect(mfr.name).toBe('Test Mfr');
      expect(mfr.email).toBe('test@test.com');
      expect(mfr.materials_supplied).toBe('Copper');
    });
  });

  describe('Order interface', () => {
    it('should have all required properties', () => {
      const order: Order = {
        id: 1, material_id: 1, quantity: 50,
        status: 'pending', notes: 'Urgent',
        material_name: 'Copper Wire',
        manufacturer_name: 'Test Mfr'
      };
      expect(order.material_id).toBe(1);
      expect(order.quantity).toBe(50);
      expect(order.status).toBe('pending');
      expect(order.material_name).toBe('Copper Wire');
    });
  });
});
