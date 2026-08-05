import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

const policySource = await readFile(
  resolve("web/krea2_strict_official_policy.js"),
  "utf8",
);
// IMPORTANT: Node 24 removed the default-type flag; a data URL preserves ESM on Node 18+.
const policyUrl = `data:text/javascript;base64,${Buffer.from(policySource).toString("base64")}`;

async function policyModule() {
  return import(policyUrl);
}

function nodeWithWidgets(variant, strictOfficial) {
  let dirtyCalls = 0;
  const node = {
    widgets: [
      { name: "variant", value: variant },
      { name: "strict_official", value: strictOfficial, disabled: false },
    ],
    setDirtyCanvas() {
      dirtyCalls += 1;
    },
  };
  return { node, dirtyCalls: () => dirtyCalls };
}

test("both Experimental variants force false and disable strict_official", async () => {
  const { synchronizeKrea2StrictOfficialPolicy } = await policyModule();

  for (const variant of [
    "LoRA Experimental (RAW mu)",
    "LoRA Experimental (Turbo mu)",
  ]) {
    const { node } = nodeWithWidgets(variant, true);
    assert.equal(synchronizeKrea2StrictOfficialPolicy(node), true);
    assert.equal(node.widgets[1].value, false);
    assert.equal(node.widgets[1].disabled, true);
  }
});

test("official variants re-enable the widget without forcing its value", async () => {
  const { synchronizeKrea2StrictOfficialPolicy } = await policyModule();

  for (const [variant, value] of [
    ["Turbo", true],
    ["RAW", false],
  ]) {
    const { node } = nodeWithWidgets(variant, value);
    node.widgets[1].disabled = true;
    assert.equal(synchronizeKrea2StrictOfficialPolicy(node), true);
    assert.equal(node.widgets[1].value, value);
    assert.equal(node.widgets[1].disabled, false);
  }
});

test("live variant changes immediately synchronize the strict widget", async () => {
  const { installKrea2StrictOfficialPolicy } = await policyModule();
  const { node, dirtyCalls } = nodeWithWidgets("Turbo", true);
  let originalCalls = 0;
  node.widgets[0].callback = () => {
    originalCalls += 1;
    return "original-result";
  };

  assert.equal(installKrea2StrictOfficialPolicy(node), true);
  assert.equal(
    node.widgets[0].callback("LoRA Experimental (Turbo mu)"),
    "original-result",
  );
  assert.equal(node.widgets[0].value, "LoRA Experimental (Turbo mu)");
  assert.equal(node.widgets[1].value, false);
  assert.equal(node.widgets[1].disabled, true);
  assert.equal(originalCalls, 1);

  node.widgets[0].callback("RAW");
  assert.equal(node.widgets[1].value, false);
  assert.equal(node.widgets[1].disabled, false);
  assert.equal(originalCalls, 2);
  assert.ok(dirtyCalls() >= 3);
});

test("missing target widgets fail closed without mutating unrelated widgets", async () => {
  const { installKrea2StrictOfficialPolicy, synchronizeKrea2StrictOfficialPolicy } =
    await policyModule();
  const unrelated = { name: "strict_official_other", value: true, disabled: false };
  const node = { widgets: [unrelated] };

  assert.equal(synchronizeKrea2StrictOfficialPolicy(node), false);
  assert.equal(installKrea2StrictOfficialPolicy(node), false);
  assert.deepEqual(unrelated, {
    name: "strict_official_other",
    value: true,
    disabled: false,
  });
});
