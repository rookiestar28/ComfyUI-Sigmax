const EXPERIMENTAL_VARIANTS = new Set([
  "LoRA Experimental (RAW mu)",
  "LoRA Experimental (Turbo mu)",
]);
const INSTALL_MARK = Symbol("Sigmax.Krea2StrictOfficialPolicy");

function targetWidgets(node) {
  if (!node || !Array.isArray(node.widgets)) return null;
  const variant = node.widgets.find((widget) => widget?.name === "variant");
  const strictOfficial = node.widgets.find(
    (widget) => widget?.name === "strict_official",
  );
  return variant && strictOfficial ? { variant, strictOfficial } : null;
}

export function synchronizeKrea2StrictOfficialPolicy(node) {
  const widgets = targetWidgets(node);
  if (!widgets) return false;

  const experimental = EXPERIMENTAL_VARIANTS.has(widgets.variant.value);
  if (experimental) widgets.strictOfficial.value = false;
  widgets.strictOfficial.disabled = experimental;
  node.setDirtyCanvas?.(true, true);
  return true;
}

export function installKrea2StrictOfficialPolicy(node) {
  const widgets = targetWidgets(node);
  if (!widgets) return false;

  if (!widgets.variant[INSTALL_MARK]) {
    const originalCallback = widgets.variant.callback;
    widgets.variant.callback = function (value, ...args) {
      const result = originalCallback?.apply(this, [value, ...args]);
      widgets.variant.value = value;
      synchronizeKrea2StrictOfficialPolicy(node);
      return result;
    };
    widgets.variant[INSTALL_MARK] = true;
  }
  return synchronizeKrea2StrictOfficialPolicy(node);
}
