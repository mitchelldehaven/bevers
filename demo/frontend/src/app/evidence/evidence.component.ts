import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, ComponentFactoryResolver, Input, OnChanges, OnInit, SimpleChanges } from '@angular/core';
import { firstValueFrom, lastValueFrom } from 'rxjs';


interface WikipediaResponse {
  query: Record<string, any>
}

interface PageInterface {
  pages: PageIdInterface;
}

interface PageIdInterface {
  [pageid: string]: Map<string, any>;
}

@Component({
  selector: 'app-evidence',
  templateUrl: './evidence.component.html',
  styleUrls: ['./evidence.component.css']
})
export class EvidenceComponent implements OnInit{

  @Input() title!: string;
  @Input() line!: string;
  @Input() score!: number;
  image_url!: string;

  constructor(public http: HttpClient) { }

  async ngOnInit(): Promise<void> {
    console.log(this.title);
    if (this.title && this.line){
      await this.get_wiki_image_url();
    }
  }


  async get_wiki_image_url(): Promise<void> {
    const headers = new HttpHeaders()
      .set('Content-Type', 'application/json')
      .set('Access-Control-Allow-Origin', '*')
    const params = {
      action: "query",
      prop: "pageimages",
      format: "json",
      piprop: "original",
      titles: this.title.replace(" ", "_"),
      origin: "*",
    };
    const wikipedia_api_url = "https://en.wikipedia.org/w/api.php"
    const post_data$ = this.http.get<WikipediaResponse>(wikipedia_api_url, { params: params });
    const response: WikipediaResponse = await lastValueFrom(post_data$);
    const pages = response.query['pages'];
    const pageid = Object.keys(pages)[0]
    if (Number(pageid) === -1) {
      return
    }
    const page_info = pages[pageid];
    if (Object.keys(page_info).indexOf("original") > -1){
      const image_info = page_info["original"];
      this.image_url = image_info["source"];
    }
    // console.log(response.query['pages']['']);
    // console.log(response);
    // console.log(response.query)
    // console.log(response.query.size)
    // console.log(response.constructor.name);
    // for (const key of response.query){
    //   console.log(key);
    // }
    // this.image_url = response.
  }

}
