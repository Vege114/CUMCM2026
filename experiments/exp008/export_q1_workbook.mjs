import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

// Execute with the bundled runtime; see --inspect for template-only verification.
const [root, mode, outputDir, payloadPath] = process.argv.slice(2);
if (!root || !["--inspect", "--smoke", "--export"].includes(mode) || !outputDir) {
  throw new Error("Usage: node export_q1_workbook.mjs ROOT --inspect|--smoke|--export OUTPUT_DIR [REPORT_PAYLOAD_JSON]");
}
await fs.mkdir(outputDir, { recursive: true });
const template = path.join(root, "data/templates/result1.xlsx");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(template));
console.log((await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 1500 })).ndjson);
const sheets = ["计划购电量", "充放电量"];
const views = [[sheets[0], "A1:B12", "purchase-first"], [sheets[0], "A68:B78", "purchase-middle"],
  [sheets[0], "A135:B145", "purchase-last"], [sheets[1], "A1:E7", "battery"]];
const sha = async p => createHash("sha256").update(await fs.readFile(p)).digest("hex");
const finite = value => typeof value === "number" && Number.isFinite(value);
const near = (actual, expected, label) => assert.ok(Math.abs(actual - expected) < 1e-6, `${label}: ${actual} vs ${expected}`);
const time = slot => `${String(Math.floor(slot / 6)).padStart(2, "0")}:${String(slot % 6 * 10).padStart(2, "0")}`;
let input;
if (mode !== "--inspect") {
  assert.ok(payloadPath, "An explicit report payload is required");
  input = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  if (mode === "--export") {
    assert.equal(input.mode, "final", "Official export requires a final report payload");
    const accepted = input.finalization_basis?.user_accepted_current_verified_result === true
      && input.finalization_basis.q2_archive_sha256 === input.scenarios?.["2"]?.archive?.sha256;
    assert.ok(input.scenarios?.["2"]?.verification?.goal?.passed === true || accepted, "Q2 requires the original goal gate or explicit user acceptance of the same verified archive");
    assert.equal(input.scenarios["2"].verification.passed, true);
    assert.ok(input.scenarios["2"].fees.total_cost_yuan <= (accepted ? 1 : 0.92) * 14066257.477256786 + 1e-6);
    assert.equal(input.missing_items.filter(r => r.blocks_final_payload).length, 0);
  }
  const q1 = input.q1;
  assert.equal(await sha(q1.archive.path), q1.archive.sha256, "Q1 source archive changed after payload preparation");
  assert.equal(q1.verification.passed, true);
  assert.equal(q1.final_trajectory_stage, 2);
  const data = q1.workbook_data;
  assert.equal(data.plan_rows.length, 144);
  assert.equal(data.battery_rows.length, 6);
  data.plan_rows.forEach((row, t) => {
    assert.equal(row.length, 2);
    assert.equal(row[0], `${time(t)}-${time(t + 1)}`);
    assert.ok(finite(row[1]) && row[1] >= -1e-6);
    near(row[1], q1.trajectory.purchase_kwh[t], `Purchase slot ${t}`);
  });
  data.battery_rows.forEach((row, b) => {
    assert.equal(row.length, 5);
    assert.equal(row[0], `${time(b * 24)}-${time((b + 1) * 24)}`);
    for (const [col, key] of [[1, "charge_kwh"], [2, "discharge_kwh"]]) {
      assert.ok(finite(row[col]) && row[col] >= -1e-6);
      near(row[col], q1.trajectory[key].slice(b * 24, (b + 1) * 24).reduce((a, v) => a + v, 0), `${key} block ${b}`);
    }
    assert.equal(row[3], b === 0 ? "0:00" : b === 1 ? "24:00" : null);
    if (b < 2) near(row[4], q1.trajectory.soc_kwh[b === 0 ? 0 : 144], `SOC ${b}`);
    else assert.equal(row[4], null);
  });
  const plan = workbook.worksheets.getItem(sheets[0]);
  const battery = workbook.worksheets.getItem(sheets[1]);
  plan.getRange("A2:B145").values = data.plan_rows;
  battery.getRange("A2:E7").values = data.battery_rows;
  plan.getRange("B2:B145").setNumberFormat("0.0000");
  battery.getRange("B2:C7").setNumberFormat("0.0000");
  battery.getRange("E2:E7").setNumberFormat("0.0000");
  plan.getRange("A1:B145").format.columnWidthPx = 155;
  battery.getRange("A1:E7").format.columnWidthPx = 155;
  plan.getRange("B2:B145").format.horizontalAlignment = "right";
  battery.getRange("B2:C7").format.horizontalAlignment = "right";
  battery.getRange("E2:E7").format.horizontalAlignment = "right";
  plan.freezePanes.freezeRows(1);
  workbook.recalculate();
}
for (const [name, range, label] of views) {
  console.log((await workbook.inspect({ kind: "table", range: `${name}!${range}`, include: "values,formulas", maxChars: 1800, tableMaxRows: 12, tableMaxCols: 5 })).ndjson);
  const preview = await workbook.render({ sheetName: name, range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(outputDir, `${mode === "--inspect" ? "template" : "preview"}-${label}.png`), new Uint8Array(await preview.arrayBuffer()));
}
if (input) {
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 30 }, summary: "Q1 output error scan" });
  console.log(errors.ndjson);
  const output = path.join(outputDir, mode === "--smoke" ? "q1_adapter_smoke.xlsx" : "result1.xlsx");
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(output);
  const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(output));
  const data = input.q1.workbook_data;
  for (const [name, range, expected] of [[sheets[0], "A2:B145", data.plan_rows], [sheets[1], "A2:E7", data.battery_rows]]) {
    const actual = saved.worksheets.getItem(name).getRange(range).values;
    assert.equal(actual.length, expected.length);
    actual.forEach((row, i) => row.forEach((value, j) => {
      if (finite(expected[i][j])) near(value, expected[i][j], `${name} row ${i + 2} col ${j + 1}`);
      else assert.equal(value ?? null, expected[i][j]);
    }));
  }
  const audit = { mode, status: mode === "--smoke" ? "temporary adapter verification, not a selected final result" : "Q1 workbook exported from final payload",
    template_sha256: await sha(template), input_sha256: await sha(payloadPath), output_sha256: await sha(output),
    source: input.q1.archive, plan_rows: 144, four_hour_blocks: 6, units: "kWh; Q1 source powers converted once in report_payload",
    all_exported_cells_reimported_and_compared: true, first_interval: data.plan_rows[0][0], last_interval: data.plan_rows[143][0],
    total_purchase_kwh: data.plan_rows.reduce((sum, row) => sum + row[1], 0),
    total_charge_kwh: data.battery_rows.reduce((sum, row) => sum + row[1], 0),
    total_discharge_kwh: data.battery_rows.reduce((sum, row) => sum + row[2], 0), formula_error_scan: errors.ndjson };
  await fs.writeFile(path.join(outputDir, "export-audit.json"), JSON.stringify(audit, null, 2) + "\n");
  console.log(JSON.stringify({ output, ...audit }));
}
