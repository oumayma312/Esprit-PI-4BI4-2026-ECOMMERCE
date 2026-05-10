import { ComponentFixture, TestBed } from '@angular/core/testing';

import { IaModuleComponent } from './ia-module.component';

describe('IaModuleComponent', () => {
  let component: IaModuleComponent;
  let fixture: ComponentFixture<IaModuleComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [IaModuleComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(IaModuleComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
