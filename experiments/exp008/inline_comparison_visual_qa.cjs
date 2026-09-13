const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

async function run() {
  const base = path.resolve('reports/experiments/exp008/evidence');
  const browser = await chromium.launch({ headless: true });
  const observations = [];
  for (const [width, theme] of [[736,'light'],[360,'light'],[736,'dark'],[360,'dark']]) {
    const page = await browser.newPage({viewport:{width:width+32,height:1800},colorScheme:theme,deviceScaleFactor:1});
    const errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('file://'+path.join(base,'comparison.preview.html'));
    const frame=page.frames().find(f=>f!==page.mainFrame());
    await frame.waitForSelector('#exp008-history-inline[data-rendered="true"]',{timeout:20000});
    await frame.waitForFunction(()=>[...document.querySelectorAll('svg.history-plot')].every(x=>x.dataset.labelConflicts!==undefined));
    const initial=await frame.evaluate(()=>({height:document.getElementById('exp008-history-inline').scrollHeight}));
    await page.locator('iframe').evaluate((el,h)=>el.style.height=h+'px',initial.height+20);
    const layout=await frame.evaluate(()=>{
      const root=document.getElementById('exp008-history-inline'), box=root.getBoundingClientRect();
      return {rootWidth:box.width,rootScrollWidth:root.scrollWidth,rootHeight:box.height,
        svg:[...root.querySelectorAll('svg.history-plot')].map(svg=>({label:svg.getAttribute('aria-label'),width:svg.getBoundingClientRect().width,
          xTickCount:svg.querySelectorAll('line.plot-grid').length,
          conflicts:JSON.parse(svg.dataset.labelConflicts||'[]'),outside:JSON.parse(svg.dataset.labelsOutside||'[]')})),
        tables:[...root.querySelectorAll('table')].map(t=>({width:t.getBoundingClientRect().width,scrollWidth:t.scrollWidth})),
        visibleTitles:[...root.querySelectorAll('h3')].map(t=>t.textContent),
        missingValues:[...root.querySelectorAll('text')].filter(t=>t.textContent.includes('缺失')).map(t=>t.textContent),
        bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,
        sourceData:JSON.parse(document.getElementById('exp008-comparison-data').textContent)};
    });
    const screenshot=path.join(base,`comparison-${width}-${theme}.png`);
    await page.locator('iframe').screenshot({path:screenshot});
    observations.push({width,theme,errors,screenshot,layout});
    await page.close();
  }
  await browser.close();
  const result={complete:true,observations};
  fs.writeFileSync(path.join(base,'inline_comparison_browser_qa.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(observations.map(r=>({width:r.width,theme:r.theme,rootWidth:r.layout.rootWidth,errors:r.errors,
    overflow:r.layout.bodyOverflow,conflicts:r.layout.svg.filter(s=>s.conflicts.length||s.outside.length)})),null,2));
}
run().catch(e=>{console.error(e);process.exitCode=1;});
