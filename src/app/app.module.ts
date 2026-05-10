import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { LoginComponent } from './login/login.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { SidebarComponent } from './sidebar/sidebar.component';
import { FaceIdComponent } from './face-id/face-id.component';
import { MlComponent } from './ml/ml.component';
import { ChatbotWidgetComponent } from './chatbot-widget/chatbot-widget.component';
import { LandingComponent } from './landing/landing.component';
import { AiInsightsComponent } from './ai-insights/ai-insights.component';
import { IaModuleComponent } from './ia-module/ia-module.component';
import { DecisionSupportComponent } from './decision-support/decision-support.component';

@NgModule({
  declarations: [
    AppComponent,
    LoginComponent,
    DashboardComponent,
    SidebarComponent,
    FaceIdComponent,
    MlComponent,
    ChatbotWidgetComponent,
    LandingComponent,
    AiInsightsComponent,
    IaModuleComponent,
    DecisionSupportComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
