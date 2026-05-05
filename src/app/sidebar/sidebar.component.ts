import { Component, OnInit } from '@angular/core';
import { AuthService, UserAccount } from '../services/auth.service';

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})
export class SidebarComponent implements OnInit {
  currentUser: UserAccount | null = null;

  constructor(public auth: AuthService) {}

  ngOnInit(): void {
    this.currentUser = this.auth.current();
  }

  logout() {
    this.auth.logout();
  }

  canAccessDashboard(role: string) {
    return this.currentUser ? this.auth.allowedDashboardRoles(this.currentUser.role).includes(role) : false;
  }
}
