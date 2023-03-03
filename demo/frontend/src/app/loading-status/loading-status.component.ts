import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-loading-status',
  templateUrl: './loading-status.component.html',
  styleUrls: ['./loading-status.component.css']
})
export class LoadingStatusComponent implements OnInit {
  @Input() getting_evidence: boolean = false;
  @Input() getting_top5: boolean = false;
  @Input() getting_pred: boolean = false;

  constructor() { }

  ngOnInit(): void {
  }

}
