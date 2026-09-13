import fs from 'node:fs/promises';
import path from 'node:path';
import { importRuntimeModule } from 'file:///C:/Users/邱志烨/.codex/plugins/cache/openai-primary-runtime/presentations/26.818.11542/skills/presentations/container_tools/runtime_helpers.mjs';
const { FileBlob, PresentationFile } = await importRuntimeModule('@oai/artifact-tool');
const source = 'D:/junk mass/小乱七八糟/数学建模/26国赛总赛/26国赛预赛/数学建模论文精美流程图中英文模板PPT！.pptx';
const deck = await PresentationFile.importPptx(await FileBlob.load(source));
const out = path.resolve('tmp/analysis-edit/reference');
await fs.mkdir(out, {recursive:true});
for (const [i,slide] of deck.slides.items.entries()) {
  const blob = await deck.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
  console.log(`Rendered ${i+1}`);
}
