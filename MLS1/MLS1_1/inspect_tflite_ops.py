# inspect_tflite_ops.py
import sys
from tensorflow.lite.python import schema_py_generated as schema

if len(sys.argv) < 2:
    print("Usage: python inspect_tflite_ops.py <model.tflite>")
    print("No model file provided, using 'mnist_model.tflite' as default.")
    fn = "mnist_model.tflite"
else:
    fn = sys.argv[1]
with open(fn, "rb") as f:
    buf = f.read()
m = schema.Model.GetRootAsModel(buf, 0)
# list operator codes
codes = []
for i in range(m.OperatorCodesLength()):
    op = m.OperatorCodes(i)
    bc = op.BuiltinCode()
    # builtin code -> name mapping defined in schema (or print value)
    print("opcode index", i, "builtin code:", bc)
# list all subgraphs' operators with operand names (more informative)
for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    print("Subgraph", si, "name:", sg.Name().decode() if sg.Name() else "")
    for oi in range(sg.OperatorsLength()):
        op = sg.Operators(oi)
        opcodeIndex = op.OpcodeIndex()
        opcode = m.OperatorCodes(opcodeIndex)
        builtin = opcode.BuiltinCode()
        print("  operator", oi, "opcodeIndex", opcodeIndex, "builtinCode", builtin)
