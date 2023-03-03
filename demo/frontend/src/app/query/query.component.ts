import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { lastValueFrom, firstValueFrom} from "rxjs"
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';
import { ToastrService } from 'ngx-toastr';

interface Evidence {
  title: string;
  line: string;
  score: number;
}

interface SentenceResponse {
  evidence_list: Evidence[];
  evidence_ids: number[];
}

interface Top5Response {
  top5_evidence: Evidence[];
  top5_ids: number[][];
  top5_scores: number[];
}

interface PredictionResponse {
  prediction_vector: number[];
  prediction: string;
  prediction_num: number;
}

@Component({
  selector: 'app-query',
  templateUrl: './query.component.html',
  styleUrls: ['./query.component.css']
})
export class QueryComponent implements OnInit {

  api_url: string = "http://localhost:5000/";
  claim!: string;
  prev_claim!: string;
  evidence_list!: Evidence[];
  evidence_ids!: number[];
  top5_evidence!: Evidence[];
  top5_evidence_scores!: number[];
  top5_evidence_ids!: number[][];
  prediction_vector!: number[];
  prediction_num!: number;
  prediction!: string;
  getting_evidence!: boolean;
  getting_top5!: boolean;
  getting_pred!: boolean;
  querying!: boolean;

  constructor(public http: HttpClient, private modalService: NgbModal, private toastr: ToastrService) { }
  
  open(content: any) {
    this.modalService.open(content);
  }

  ngOnInit(): void {
    this.claim = "";
    this.prev_claim = "";
    this.evidence_list = [];
    this.top5_evidence = [];
    this.top5_evidence_scores = [];
    this.prediction_vector = [];
    this.prediction_num= -1;
    this.prediction = "";
    this.getting_evidence = false;
    this.getting_top5 = false;
    this.getting_pred = false;
    this.querying = false;
  }

  async get_candidate_sentences(): Promise<void> {
    this.getting_evidence = true;
    const params = {claim: this.claim};
    const get_data$ = this.http.get<SentenceResponse>(this.api_url + "api/get_candidate_sentences", { params });
    const response: SentenceResponse = await firstValueFrom(get_data$);
    console.log(response);
    if (this.querying) { 
      this.evidence_list = response.evidence_list;
      this.evidence_ids = response.evidence_ids;
      this.getting_evidence = false;
    }
  }


  async get_top5_sentences(): Promise<void> {
    this.getting_top5 = true;
    const params = {claim: this.claim, sentences: this.evidence_list, sentence_ids: this.evidence_ids};
    const post_data$ = this.http.post<Top5Response>(this.api_url + "api/get_top5_sentences", { params });
    const response: Top5Response = await lastValueFrom(post_data$);
    console.log(response);
    if (this.querying){
      this.top5_evidence = response.top5_evidence;
      this.top5_evidence_scores = response.top5_scores;
      this.top5_evidence.forEach((value, index) => {
        value.score = this.top5_evidence_scores[index];
      })
      this.top5_evidence_ids = response.top5_ids;
      this.getting_top5 = false;
    }
  }


  async get_prediction(): Promise<void> {
    this.getting_pred = true;
    const params = {claim: this.claim, top5_sentences: this.top5_evidence}
    const post_data$ = this.http.post<PredictionResponse>(this.api_url + "api/get_prediction", { params });
    const response: PredictionResponse = await lastValueFrom(post_data$)
    if (this.querying) { 
      this.prediction_vector = response.prediction_vector;
      this.prediction = response.prediction;
      this.prediction_num = response.prediction_num;
      this.getting_pred = false;
    }
  }


  hard_reset(): void {
    this.ngOnInit();
  }
  
  reset(hard_reset=true): void {
    if (hard_reset) {
      this.hard_reset();
    } else {
      this.prev_claim = "";
      this.evidence_list = [];
      this.top5_evidence = [];
      this.top5_evidence_scores = [];
      this.prediction_vector = [];
      this.prediction_num= -1;
      this.prediction = "";
    }
  }


  async verify(): Promise<void> {
    if (this.claim != this.prev_claim) {
      this.reset(false);
      this.querying = true;
      await this.get_candidate_sentences();
      if (this.evidence_list.length === 0) {
        this.querying = false;
        this.toastr.error("No documents retrieved. In this pared down version, named entity based claims ('George Washington was a president') work well.");
      }
      await this.get_top5_sentences();
      await this.get_prediction();
      this.querying = false;
    }
    this.prev_claim = this.claim;
  }

  get_loading_status(): string {
    if (this.getting_evidence) {
      return "Retrieving Evidence";
    } else if (this.getting_top5) {
      return `Getting Top 5 Evidence from ${this.evidence_list.length} Sentences (this can take up to 30-60 seconds)`;
    } else if (this.getting_pred) {
      return "Getting Prediction";
    }
    return ""
  }
}
