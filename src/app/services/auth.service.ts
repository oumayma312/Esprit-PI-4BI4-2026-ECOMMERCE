import { Injectable } from '@angular/core';
import { Router } from '@angular/router';

export interface UserAccount { email: string; password: string; role: string }

@Injectable({ providedIn: 'root' })
export class AuthService {
  private accounts: UserAccount[] = [
    { email: 'ceo@gmail.com', password: 'ceo', role: 'ceo' },
    { email: 'finance@gmail.com', password: 'finance', role: 'finance' },
    { email: 'sales@gmail.com', password: 'sales', role: 'sales' }
  ];

  private dashboardAccess: Record<string, string[]> = {
    ceo: ['ceo', 'sales', 'finance'],
    sales: ['sales'],
    finance: ['finance']
  };

  constructor(private router: Router) {}

  login(email: string, password: string): boolean {
    const found = this.accounts.find(a => a.email === email && a.password === password);
    if (found) {
      localStorage.setItem('pi_user', JSON.stringify(found));
      this.router.navigate(['/dashboard', found.role]);
      return true;
    }
    return false;
  }

  logout() {
    localStorage.removeItem('pi_user');
    this.router.navigate(['/login']);
  }

  current() : UserAccount | null {
    const raw = localStorage.getItem('pi_user');
    return raw ? JSON.parse(raw) : null;
  }

  allowedDashboardRoles(role: string) {
    return this.dashboardAccess[role] || [];
  }

  listAccounts(): UserAccount[] {
    return [...this.accounts];
  }

  // helper for FaceID flow to lookup by role/email
  findByEmail(email: string) {
    return this.accounts.find(a => a.email === email) || null;
  }
}
