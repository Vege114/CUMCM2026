const fs=require('fs');
const path=require('path');
const {chromium}=require('playwright');

async function main(){
  const out=path.resolve('reports/experiments/exp009/evidence/inline');
  const browser=await chromium.launch({headless:true});const runs=[];
  for(const [width,theme] of [[736,'light'],[360,'light'],[736,'dark'],[360,'dark']]){
    const page=await browser.newPage({viewport:{width:width+32,height:1800},colorScheme:theme,deviceScaleFactor:1});
    const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('file://'+path.join(out,'comparison.preview.html'));
    const frame=page.frames().find(f=>f!==page.mainFrame());
    await frame.waitForSelector('#exp009-control-inline[data-rendered="true"]',{timeout:30000});
    await frame.waitForFunction(()=>[...document.querySelectorAll('svg.exp009-plot')].every(x=>x.dataset.labelConflicts!==undefined));
    const h=await frame.locator('#exp009-control-inline').evaluate(el=>el.scrollHeight);
    await page.locator('iframe').evaluate((el,height)=>el.style.height=height+20+'px',h);
    const layout=await frame.evaluate(()=>{
      const r=document.getElementById('exp009-control-inline');return {width:r.clientWidth,scrollWidth:r.scrollWidth,height:r.scrollHeight,
        overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,
        svgs:[...r.querySelectorAll('svg')].map(s=>({name:s.getAttribute('aria-label'),width:s.getBoundingClientRect().width,
          conflicts:JSON.parse(s.dataset.labelConflicts),outside:JSON.parse(s.dataset.labelsOutside),
          axisTitles:[...s.querySelectorAll('text.axis-title')].map(t=>({axis:t.dataset.axis,label:t.textContent})),
          fullPowerCurves:[...s.querySelectorAll('.power-line')].map(p=>({series:p.dataset.series,points:Number(p.dataset.pointCount),
            pathCommandPoints:(p.getAttribute('d').match(/[ML]/g)||[]).length}))})),
        essentialTitles:[...r.querySelectorAll('h2,h3')].map(x=>x.textContent)};
    });
    const screenshot=path.join(out,`comparison-${width}-${theme}.png`);
    await page.locator('iframe').screenshot({path:screenshot});
    const overlay=frame.locator('[data-chart-hover-overlay]').first();const bounds=await overlay.boundingBox();
    await page.mouse.move(bounds.x+bounds.width*.417,bounds.y+bounds.height*.3);
    const before=await frame.evaluate(()=>{
      const t=document.querySelector('[role="tooltip"]');const p=document.querySelector('[data-plot="power-0"]');
      return {tooltipVisible:!t.hidden,rows:t.querySelectorAll('[data-tooltip-series]').length,
        guideX:Number(p.querySelector('[data-chart-hover-guide]').getAttribute('x1')),
        markerX:[...p.querySelectorAll('[data-chart-hover-marker]')].map(x=>Number(x.getAttribute('cx')))};
    });
    await frame.locator('[data-version="exp008"]').click();
    const toggled=await frame.evaluate(()=>({hidden:document.querySelector('[data-version="exp008"]').getAttribute('aria-pressed')==='false',
      oldPowerLines:document.querySelectorAll('.power-line[data-series="exp008"]').length,
      newPowerLines:document.querySelectorAll('.power-line[data-series="exp009"]').length}));
    const changedBounds=await overlay.boundingBox();await page.mouse.move(changedBounds.x+changedBounds.width*.417,changedBounds.y+changedBounds.height*.3);
    const after=await frame.evaluate(()=>({rows:document.querySelector('[role="tooltip"]').querySelectorAll('[data-tooltip-series]').length,
      names:[...document.querySelectorAll('[role="tooltip"] [data-tooltip-series]')].map(x=>x.dataset.tooltipSeries)}));
    await overlay.click({position:{x:changedBounds.width*.417,y:changedBounds.height*.3}});
    const pinned=await frame.locator('[data-selection]').textContent();
    runs.push({width,theme,errors,layout,screenshot,interaction:{before,toggled,after,pinned}});
    await page.close();
  }
  await browser.close();
  const passed=runs.every(r=>!r.errors.length&&!r.layout.overflow&&r.layout.width===r.width&&r.layout.scrollWidth===r.width
    &&r.layout.svgs.every(s=>!s.conflicts.length&&!s.outside.length&&s.axisTitles.length===2&&s.fullPowerCurves.every(c=>c.points===144&&c.pathCommandPoints===144))
    &&r.interaction.before.tooltipVisible&&r.interaction.before.rows===2&&r.interaction.before.markerX.every(x=>Math.abs(x-r.interaction.before.guideX)<1e-9)
    &&r.interaction.toggled.hidden&&r.interaction.toggled.oldPowerLines===0&&r.interaction.toggled.newPowerLines===4
    &&r.interaction.after.rows===1&&r.interaction.after.names[0]==='exp009'&&r.interaction.pinned.includes('已固定'));
  fs.writeFileSync(path.join(out,'browser_qa.json'),JSON.stringify({passed,runs},null,2)+'\n');
  console.log(JSON.stringify({passed,runs:runs.map(r=>({width:r.width,theme:r.theme,errors:r.errors,overflow:r.layout.overflow,
    badLabels:r.layout.svgs.filter(s=>s.conflicts.length||s.outside.length),interaction:r.interaction}))},null,2));
  if(!passed)process.exitCode=1;
}
main().catch(e=>{console.error(e);process.exitCode=1;});
