import { app } from "../../scripts/app.js";
import {
  installKrea2StrictOfficialPolicy,
  synchronizeKrea2StrictOfficialPolicy,
} from "./krea2_strict_official_policy.js";

app.registerExtension({
  name: "Sigmax.Krea2StrictOfficialPolicy",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Sigmax.Krea2SigmaScheduler") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      installKrea2StrictOfficialPolicy(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      synchronizeKrea2StrictOfficialPolicy(this);
      return result;
    };
  },
});
