import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.argv[2];
if (!root) throw new Error('Pass repository root');
const report = path.join(root, 'reports/experiments/exp009');
const out = path.join(report, 'evidence/report_browser');
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ headless: true });
const results = [];
for (const width of [1440, 390]) {
  const page = await browser.newPage({ viewport: { width, height: 1000 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  const requests = [];
  page.on('request', r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
  await page.goto(pathToFileURL(path.join(report, 'report.html')).href, { waitUntil: 'load' });
  const summary = await page.evaluate(() => ({
    sections: [...document.querySelectorAll('main h2')].map(e => e.textContent),
    images: [...document.images].length,
    failedImages: [...document.images].filter(i => !i.complete || !i.naturalWidth).map(i => i.alt),
    horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
    links: [...document.querySelectorAll('a')].map(a => a.getAttribute('href')),
  }));
  await page.screenshot({ path: path.join(out, `report-top-${width}.png`) });
  const headings = page.locator('main h2');
  for (const i of [3, 5, 6]) {
    await headings.nth(i).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(out, `report-section-${i+1}-${width}.png`) });
  }
  await page.goto(pathToFileURL(path.join(report, 'specified_dates.html')).href);
  const specified = await page.evaluate(() => ({ failedImages: [...document.images].filter(i => !i.complete || !i.naturalWidth).length, horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth, dailyTotalCount:(document.body.innerText.match(/全天总购电量/g)||[]).length }));
  await page.screenshot({ path: path.join(out, `specified-top-${width}.png`) });
  results.push({ width, errors, externalRequests: requests, ...summary, specified });
  await page.close();
}
await browser.close();
const passed = results.every(r => r.sections.length === 8 && !r.errors.length && !r.externalRequests.length && !r.failedImages.length && !r.horizontalOverflow && !r.specified.horizontalOverflow && r.specified.dailyTotalCount === 17);
fs.writeFileSync(path.join(out, 'browser_checks.json'), JSON.stringify({ passed, results, visualReview: 'Screenshots require separate human-visible inspection; this script checks runtime and geometry only.' }, null, 2)+'\n');
console.log(JSON.stringify({passed, screens:results.map(r=>({width:r.width,sections:r.sections.length,images:r.images,overflow:r.horizontalOverflow, dailyTotals:r.specified.dailyTotalCount}))},null,2));
if (!passed) process.exitCode = 1;
