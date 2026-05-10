import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { MlComponent } from './ml/ml.component';
import { LandingComponent } from './landing/landing.component';
import { AiInsightsComponent } from './ai-insights/ai-insights.component';
import { DecisionSupportComponent } from './decision-support/decision-support.component';

const routes: Routes = [
  { path: '', component: LandingComponent },
  { path: 'login', component: LoginComponent },
  { path: 'dashboard/:role', component: DashboardComponent },
  { path: 'ai-insights', component: AiInsightsComponent },
  { path: 'ml', component: MlComponent },
  { path: 'decision-support', component: DecisionSupportComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes, {
    anchorScrolling: 'enabled',
    scrollPositionRestoration: 'enabled'
  })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
