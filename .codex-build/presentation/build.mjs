import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';
import { finalizePresentation, applyPresentationChartFont } from '/Users/Apple/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations/container_tools/artifact_tool_utils.mjs';

const ROOT='/Users/Apple/Documents/MinneMUDAC-2025';
const BUILD=path.join(ROOT,'.codex-build/presentation');
const SKILL='/Users/Apple/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations';
const PY='/Users/Apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const FINAL=path.join(ROOT,'output/MinneMUDAC_2025_Updated.pptx');
const C={bg:'#F5F6F2',ink:'#16251E',muted:'#52645A',green:'#007D55',bright:'#00D58B',dark:'#101B16',white:'#FFFFFF',line:'#D8E0D9',amber:'#975822'};
const P=Presentation.create({slideSize:{width:1280,height:720}});
const font='Arial';
const analysis=JSON.parse(await fs.readFile(path.join(BUILD,'analysis.json'),'utf8'));
const sources={
 data:'Repository: Data Sources.docx; MUDAC/DataDictionary.xlsx; MUDAC/Training_with_Life_Events.xlsx; MUDAC/Test_Truncated.xlsx; MUDAC/Novice.xlsx.',
 events:'Repository: MUDAC/Apr3TrainRun1LLMFlattened1.xlsx. Aggregates calculated September 30, 2026. Count a row when the event column is numeric and greater than zero. Repeated events for one match remain separate rows. These are machine-extracted labels, without a measured annotation accuracy in the repository.',
 models:'Repository: MUDAC/Completed/ML Training Normal.ipynb, saved output cell 5 and source cells 0–5; MUDAC/minnie_mudac.ipynb, saved output and source cell 1. Match Length in Months from MUDAC/DataDictionary.xlsx. Results are saved historical notebook outputs, not new training runs.',
 code:'Repository: Grok Prompt Pipeline/grok_prompt_processor.py; ML Pipeline/advanced_ml_training.py. Static review plus local synthetic checks of JSON flattening and error recording. No API calls or retraining.'
};
function text(s,t,x,y,w,h,size=25,color=C.ink,bold=false){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=t;sh.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none',wrap:'square',verticalAlignment:'top',insets:{top:0,bottom:0,left:0,right:0}};return sh;
}
function slide(title,notes,dark=false){
 const s=P.slides.add();s.background.fill=dark?C.dark:C.bg;
 if(title)text(s,title,64,50,1152,108,46,dark?C.white:C.ink,true);
 text(s,String(P.slides.items.length).padStart(2,'0'),1170,676,50,22,16,dark?'#A5B7AA':C.muted);
 s.speakerNotes.textFrame.setText(notes);return s;
}
function caption(s,t,y=640,dark=false){text(s,t,64,y,1100,39,19,dark?'#BCCDC2':C.muted);}
function row(s,label,body,y,{x=64,lw=295,w=1130,color=C.green,size=26}={}){
 text(s,label,x,y,lw,80,size,color,true);text(s,body,x+lw+35,y,w-lw-35,100,size,C.ink);
}
function table(s,values,{x=64,y=184,w=1152,h=388,widths,fontSize=24}={}){
 const tb=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});
 tb.borders.assign({fill:C.line,width:1,style:'solid'});
 const all=tb.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length});
 all.assign({textStyle:{typeface:font,fontSize,color:C.ink},fill:C.bg,margins:{left:15,right:15,top:12,bottom:10},anchor:'center'});
 tb.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:C.ink,textStyle:{typeface:font,fontSize,bold:true,color:C.white}});
 return tb;
}
function bars(s,categories,values,{x=64,y=175,w=1152,h=430,title,color=C.green,max,unit=1000,number='0',horizontal=true}={}){
 const ch=s.charts.add('bar',{
  position:{left:x,top:y,width:w,height:h},categories:horizontal?[...categories].reverse():categories,
  series:[{name:title??'Count',values:horizontal?[...values].reverse():values,fill:color,valuesFormatCode:number}],
  barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:65},hasLegend:false,
  chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0},
  xAxis:{textStyle:{fontSize:22,fill:C.muted},line:{fill:'none',width:0},majorGridlines:null},
  yAxis:{min:0,...(max?{max}:{}),majorUnit:unit,numberFormatCode:number,textStyle:{fontSize:21,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1},title:{text:title??'Rows with an event',textStyle:{fontSize:22,fill:C.muted}}},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:22,bold:true,fill:C.ink}}
 });applyPresentationChartFont(ch,{fontFamily:font});return ch;
}
async function image(s,filename,x,y,w,h,alt){
 const data=await fs.readFile(path.join(ROOT,filename));
 s.images.add({blob:new Uint8Array(data),contentType:filename.endsWith('.jpg')?'image/jpeg':'image/png',position:{left:x,top:y,width:w,height:h},fit:'contain',alt});
}

// 1
{
 const s=slide('', 'Project: MinneMUDAC 2025, Team U37. Updated September 30, 2026 for project judges and Big Brothers Big Sisters stakeholders. Photo: repository asset BI Dashboard/Powerpoint/bbb pic.jpg. The image illustrates BBB branding and does not establish participant identity or project outcomes.',true);
 text(s,'Mentorship\nmatch analytics',64,92,690,188,70,C.white,true);
 text(s,'Big Brothers Big Sisters',68,310,650,46,30,C.bright);
 text(s,'Support notes, match duration\nand staff decision support',68,377,630,105,30,C.white);
 await image(s,'BI Dashboard/Powerpoint/bbb pic.jpg',738,150,478,397,'Repository photograph of young people wearing Big Brothers Big Sisters shirts');
 text(s,'Team U37\nMinneMUDAC 2025',68,573,520,77,24,'#BCCDC2');
 text(s,'Updated September 30, 2026',738,584,478,50,22,'#BCCDC2');
}
// 2
{
 const s=slide('Project contribution',`${sources.data}\n${sources.events}\n${sources.code}\nThe project aims to reduce manual reading effort and make support signals visible. No measured staff time savings, operational intervention benefit, or event extraction accuracy is available. The summary describes implemented artifacts and a proposed use.`);
 text(s,'3,275',64,184,500,108,94,C.green,true);
 text(s,'matches in the source cohort',68,298,535,48,28);
 text(s,'Support notes contain the context\nthat structured fields can miss.',660,188,530,127,35,C.ink,true);
 row(s,'Event extraction','Grok converts narrative notes into event labels and flag counts.',408,{lw:270,size:26});
 row(s,'Match duration','Saved notebooks explore statistical models and LSTM sequences.',492,{lw:270,size:26});
 row(s,'Staff visibility','Power BI brings event trends and match context into one view.',576,{lw:270,size:26});
}
// 3
{
 const s=slide('Data and prediction task',`${sources.data}\nTraining_with_Life_Events.xlsx contains 39,345 rows. Test_Truncated.xlsx contains 2,566 rows and 300 distinct Match IDs. Novice.xlsx contains 3,275 distinct matches. The data source brief says the challenge removes recent note instances and withholds closure fields and Match Length from the test input. The saved flattened extraction contains 25,050 rows covering all 3,275 distinct match IDs. MUDAC/ML_Training_Test_Dataset.xlsx and Completed Test Dataset.xlsx each contain 1,705 processed rows for 300 matches; their names alone do not establish original training provenance.`);
 table(s,[['Artifact','Rows','Matches','Role'],['Training source','39,345','3,275','Match and note instances'],['Truncated test input','2,566','300','Partial histories to predict duration'],['Saved event extraction','25,050','3,275','LLM features for analysis']],{h:314,widths:[335,150,160,507]});
 text(s,'Target: match length in months',64,531,1145,47,31,C.green,true);
 caption(s,'Test input omits recent notes and closure fields. Active matches remain ongoing.',594);
}
// 4
{
 const s=slide('Match status and duration',`${sources.data}\nComputed from MUDAC/Novice.xlsx. One row per match, n=3,275. Stage counts: Closed 2,486, Active 774, Pending Closure 15. Closed-match duration: median 15.7 months, mean 20.7371 months, interquartile range 8.7–28.2 months. Active matches are still ongoing and should not be interpreted as final-duration observations. The chart and figures describe this repository snapshot, not current BBB operations.`);
 bars(s,['Closed','Active','Pending closure'],[2486,774,15],{x:64,y:199,w:715,h:383,title:'Matches',horizontal:false,max:3000,unit:1000});
 text(s,'15.7',862,196,340,106,88,C.green,true);
 text(s,'months\nmedian closed-match length',865,300,326,100,27);
 text(s,'8.7 to 28.2 months',860,466,348,51,30,C.ink,true);
 text(s,'Middle 50% of closed matches',864,525,343,66,25,C.muted);
 caption(s,'Source snapshot: 3,275 matches. Duration figures use the 2,486 closed matches only.');
}
// 5
{
 const s=slide('Analysis pipeline',`${sources.code}\nRepository evidence also includes MUDAC/Geo/GeoCode.ipynb, MUDAC/Geo/mn_block_group_centroids.csv, CensusACSBlockGroup(CensusACSBlockGroup).csv, notebooks under MUDAC/Completed, and BI Dashboard/PowerBi BBB Dashboard.pbix. Saved spreadsheets demonstrate intermediate work. These files do not prove the standalone Python pipeline runs from end to end without repairs. Automated calling, speech transcription, MatchForce API integration and live alerts appear as proposed capabilities in existing slides, without an implemented integration in the reviewed repository.`);
 table(s,[['Stage','Input and transformation','Output'],['Preparation','Support notes, match fields and Census context','Cleaned match and note records'],['Semantic extraction','Grok identifies events and positive or concerning signals','34 event fields and flag counts'],['Modeling','Aggregate features or dated match sequences','Exploratory duration estimates'],['Reporting','Power BI organizes trends and match context','Views for staff review']],{y:177,h:391,widths:[240,536,376],fontSize:23});
 caption(s,'Automated calls, MatchForce integration and live alerts remain proposed extensions.',613);
}
// 6
{
 const s=slide('Note extraction example',`${sources.code}\nThis is a synthetic example written for this presentation. It is not an actual participant note or a real Grok output. Event name follows EXPECTED_EVENT_COLUMNS. Counts and severity illustrate the output contract and should not be read as validated risk probabilities. The current prompt uses a list of event objects despite wording that says object. The extractor validates allowed event names and a severity range of 1–5 for present events. Dates in saved flattened artifacts are separate from the shown output contract.`);
 text(s,'Illustrative support note',64,184,525,50,29,C.green,true);
 text(s,'“The volunteer has struggled to\nreach the family. Both want to\ncontinue, and the coordinator\nwill arrange a check-in.”',64,258,543,223,31);
 text(s,'Structured output',670,184,540,50,29,C.green,true);
 text(s,'[\n  {\n    "green_flag_count": 1,\n    "red_flag_count": 1,\n    "events": {\n      "Child/Family: Lost contact with volunteer": 3\n    }\n  }\n]',670,249,538,352,22);
 caption(s,'Synthetic illustration. Event severity is an assigned label, not a calibrated probability.');
}
// 7
{
 const s=slide('Frequent extracted events',`${sources.events}\nTop six event columns: COVID impact 3,758; Match closure Discussed 2,237; Child/Family: Lost contact with volunteer 1,552; Child/Family: Moved 1,201; Child/Family: Time constraints 994; Changing Match Type 866. Each is a count of rows with a severity greater than zero out of 25,050 extracted rows, not distinct matches, closure rates, or validated causes of termination. Event fields can overlap in the same row. Closure discussions and events recorded late in a match need strict temporal treatment in predictive evaluation.`);
 bars(s,['COVID impact','Closure discussed','Family lost contact with volunteer','Family moved','Family time constraints','Match type changed'],[3758,2237,1552,1201,994,866],{y:177,h:425,max:4500,unit:1000});
 caption(s,'Counts of rows with an event across 25,050 extracted rows. Repeated events can belong to the same match.');
}
// 8
{
 const s=slide('Support priorities for staff review',`${sources.events}\nThe suggested responses are project recommendations, not measured interventions or formal BBB policy. They connect common extracted event categories with questions a coordinator could investigate. A label alone is insufficient to infer a family circumstance, assign blame or make an automated closure decision. Existing escalation procedures govern safety concerns.`);
 table(s,[['Signal in a note','Suggested coordinator response'],['Loss of contact','Confirm contact preferences and arrange a joint check-in.'],['Relocation or transport barriers','Review travel access and whether the match can continue.'],['Time constraints or loss of interest','Agree on a realistic meeting plan with the participants.'],['Closure discussion','Review the context and support an appropriate closure plan.']],{y:182,h:389,widths:[365,787],fontSize:25});
 caption(s,'A coordinator reviews the source note and context before taking action.',633);
}
// 9
{
 const s=slide('Saved model results',`${sources.models}\nRandomForestRegressor: cross-validation RMSE 2.7296, held-out test RMSE 2.8893. minnie_mudac.ipynb: five-fold mean RMSE 6.57, MAE 3.94, R² 0.8900. The experiments reference different input workbook paths, and the repository does not establish identical cohorts or observation cutoffs. ML Completed Subset.xlsx is missing. Both notebooks fit some preprocessing before splitting or cross-validation. LSTM hyperparameters are selected on one of the folds then reused across five-fold reporting. These limitations prevent a defensible head-to-head comparison or deployment accuracy claim. No saved performance output for the standalone advanced ensemble script was located.`);
 table(s,[['Experiment','Recorded result','Evaluation'],['Random Forest','RMSE 2.89 months','Held-out notebook test split'],['Random Forest','RMSE 2.73 months','Notebook cross-validation'],['Tuned LSTM','RMSE 6.57 months\nMAE 3.94 months, R² 0.89','Five-fold notebook averages']],{y:181,h:341,widths:[310,445,397],fontSize:25});
 text(s,'These experiments do not support a direct model ranking',64,562,1152,62,30,C.green,true);
 caption(s,'Different input provenance and evaluation limitations require a fresh common benchmark.',636);
}
// 10
{
 const s=slide('Validation needed before prediction use',`${sources.models}\n${sources.code}\nEvidence: preprocessing is fitted before model splitting in advanced_ml_training.py line 600 and before the five-fold loop in minnie_mudac.ipynb. The normal notebook imputes before its match-level split and aggregates max_update_days, note count and closure-discussion labels. Whole histories may reveal final duration rather than information available at an early prediction point. Advanced script retains object and numeric columns without an explicit inference-time allowlist and uses whole-match aggregates. Dataset includes Active matches whose recorded elapsed length is censored. Proposed benchmark must hold out matches and observation time, fit preprocessing only on training folds, exclude future data and closure-only fields, compare simple baselines and evaluate by program subgroup.`);
 row(s,'Observation cutoff','Use only notes and fields available when the prediction is made.',186,{lw:320,size:27});
 row(s,'Independent evaluation','Hold out matches and later periods. Fit preprocessing inside each training fold.',294,{lw:320,size:27});
 row(s,'Ongoing matches','Separate elapsed duration from final closure duration for active matches.',421,{lw:320,size:27});
 row(s,'Common benchmark','Compare against a simple baseline on the same cohort and cutoff.',540,{lw:320,size:27});
}
// 11
{
 const s=slide('Existing Power BI dashboard', 'Repository: BI Dashboard/PowerBi BBB Dashboard.pbix. Screenshot: extracted image46.png from MUDAC/Presentation/Draft Data Stompers.pptx. This is an existing saved dashboard view, not a live refreshed report. Its filters, dates, counts and chart scales may differ from the workbook aggregates elsewhere in this deck. The screenshot shows event severity trends and flag totals without participant-level records. Dual axes and temporal correlations should not be interpreted as causal relationships or intervention benefit. No Power BI refresh or external service integration was performed.');
 await image(s,'.codex-build/presentation/source-media/image46.png',64,166,905,512,'Existing Power BI screenshot of event severity trends and flag counts');
 text(s,'Event trends',1000,208,223,44,28,C.green,true);
 text(s,'Review patterns by\nperiod and program.',1000,260,219,118,25);
 text(s,'Match context',1000,422,223,44,28,C.green,true);
 text(s,'Use the dashboard\nto guide a staff\nreview of the notes.',1000,474,219,133,25);
}
// 12
{
 const s=slide('LLM cost and implementation', `Repository: README.md judge feedback: "LLM Calling is an interesting idea - but consider cost of implementations." Grok Prompt Pipeline/grok_prompt_processor.py query_grok passes accumulated conversation_history and uses max_tokens=1000. History therefore adds input tokens across successive calls within a match. The repository does not include a complete cost ledger, cached-input accounting, per-successful-record cost, or measured staff time savings. Recommendations and the formula are proposed accounting steps, not reported financial outcomes. No current model pricing is quoted.`);
 text(s,'Cost depends on note volume\nand repeated conversation context',64,184,1143,115,38,C.green,true);
 row(s,'Usage accounting','Track billed input and output tokens, retries and successful records.',344,{lw:290,size:26});
 row(s,'Processing approach','Process new notes incrementally and reuse unchanged results.',439,{lw:290,size:26});
 row(s,'Pilot comparison','Measure cost per reviewed match alongside staff review time.',534,{lw:290,size:26});
 caption(s,'Estimated API cost = input tokens × input rate + output tokens × output rate, using consistent billing units.');
}
// 13
{
 const s=slide('Competition feedback', 'Repository: README.md, Competition Results section. Team U37 Round 1 overall score 60.29/80. Reported all-team mean 57.2. Rubric scores: Creativity & Innovation 3.50, Communication of Outcomes & Team Synergy 3.10, Prediction 3.03, Impact of Important Factors 3.00, Completeness & Breadth of Outcome 2.80, Appropriateness of Analytical Methods 2.80. Source score sheet is not present, so these remain README-reported results. No percentile or ranking assertion is included. Quoted judge feedback is as recorded in the README.');
 text(s,'60.29 / 80',64,177,520,106,75,C.green,true);
 text(s,'Team U37, Round 1',69,298,500,42,26);
 text(s,'57.2',748,177,460,106,75,C.ink,true);
 text(s,'Reported all-team mean',752,298,462,42,26);
 text(s,'“LLM Calling is an interesting idea -\nbut consider cost of implementations.”',64,411,1150,128,39,C.ink,true);
 caption(s,'README-reported results. Cost measurement and evaluation rigor are priorities for the next iteration.',609);
}
// 14
{
 const s=slide('Proposed pilot', 'Recommendation based on repository review, not a scheduled or approved deployment. Suggested owners: technical team for pipeline repair and model benchmark, coordinators for label review, BBB project sponsor for pilot decision. Before any external processing of real support notes, resolve credential handling and confirm approved data handling. A staff-reviewed evaluation should measure event precision and recall on an annotated sample, review time, alert burden, cost per reviewed match and subgroup performance. A future pilot decision should follow agreed thresholds rather than thresholds invented for this deck.');
 table(s,[['Phase','Work','Evidence for the next decision'],['Repair and benchmark','Fix extraction and reproduce a common model test.','Reliable output and a valid baseline comparison'],['Staff label review','Coordinators annotate a sample of notes.','Event precision, recall and disagreement patterns'],['Limited pilot','Staff review suggested signals during normal work.','Review time, alert burden and cost per match']],{y:188,h:374,widths:[275,420,457],fontSize:24});
 caption(s,'BBB stakeholders set the success thresholds before the pilot begins.',618);
}
// 15
{
 const s=slide('Pilot decision', 'Recommended next step: a limited pilot with coordinator review after pipeline fixes and independent benchmarking. Repository evidence supports an exploratory workflow, not automated closure decisions, established early-warning performance or measured improvement in youth outcomes. Proposed extensions should be considered after reliable extraction, valid prediction testing and usage accounting.',true);
 text(s,'A limited pilot with\ncoordinator review',64,194,1100,170,62,C.white,true);
 text(s,'The repository demonstrates a way to structure support notes\nand explore match patterns.',68,413,1100,100,32,'#BCCDC2');
 text(s,'The next decision depends on extraction quality,\nindependent model results and operating cost.',68,545,1100,100,32,C.bright);
}
// 16 appendix
{
 const s=slide('Appendix: implementation priorities', `${sources.code}\nDetailed findings and exact locations are in output/Repository_Review.md. A hard-coded credential exists at grok_prompt_processor.py line 23. Its value is intentionally omitted. Default input paths reference absent files. Synthetic reproduction: flatten_json_responses given a Series of one-element response lists raises ValueError because events is not a normalized top-level column. Synthetic failure reproduction: process_row logs a simulated query failure by rebinding a local error_df and returns None, leaving the caller's error ledger empty. Sequence construction pads at the end, while tree models use X_train_seq[:, -1, :], selecting padding for shorter histories. Attention lacks a padding mask. These observations concern current code, not a new full pipeline execution.`);
 table(s,[['Priority','Required work'],['Credential handling','Revoke the exposed key and load secrets only from the environment.'],['Extraction reliability','Normalize response lists and retain failures in the returned error ledger.'],['Model correctness','Select the last observed row and mask padded attention positions.'],['Reproducibility','Repair file paths and record the cohort, cutoff, configuration and usage.']],{y:182,h:386,widths:[323,829],fontSize:24});
 caption(s,'Detailed evidence and source locations accompany the deck in Repository_Review.md.',619);
}

await fs.mkdir(path.join(BUILD,'renders'),{recursive:true});
await (await PresentationFile.exportPptx(P)).save(path.join(BUILD,'candidate.pptx'));
console.log('Exported',P.slides.items.length,'slides');
await fs.writeFile(path.join(BUILD,'presentation.proto.json'),JSON.stringify(P.toProto()));
for(let i=0;i<P.slides.items.length;i++){
 const s=P.slides.items[i];
 const png=await P.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'renders',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await s.export({format:'layout'});
 await fs.writeFile(path.join(BUILD,'renders',`slide-${String(i+1).padStart(2,'0')}.layout.json`),await layout.text());
 console.log('Rendered',i+1);
}
const result=await finalizePresentation({
 workspaceDir:ROOT,candidatePath:path.join(BUILD,'candidate.pptx'),finalPath:FINAL,pythonExecutable:PY,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,5,8,9,14,16].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:[3,5,8,9,14,16],requiredNativeChartOwnerSlides:[4,7],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(BUILD,'validation-v2.json')
});
console.log('Finalized',FINAL);
await fs.writeFile(path.join(BUILD,'finalize-result.json'),JSON.stringify(result,null,2));
// Render the finalized package, including any workbook materialization.
const reopened=await PresentationFile.importPptx(await FileBlob.load(FINAL));
for(let i=0;i<reopened.slides.items.length;i++){
 const png=await reopened.export({slide:reopened.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'renders',`final-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('Rendered finalized package');
