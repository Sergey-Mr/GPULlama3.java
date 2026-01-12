package org.beehive.gpullama3.tools;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;

/**
 * Utility class to generate OpenCL kernel for processHeadsFlashAttention.
 *
 * Run with:
 *   tornado --printKernel -m gpullama3/org.beehive.gpullama3.tools.FlashAttentionKernelGenerator
 *
 * Or if running from IDE/command line with TornadoVM:
 *   java --enable-preview @${TORNADO_SDK}/tornado-argfile \
 *     -Dtornado.opencl.accelerator.use.default=false \
 *     --add-modules ALL-SYSTEM \
 *     -cp <classpath> \
 *     org.beehive.gpullama3.tools.FlashAttentionKernelGenerator
 */
public class FlashAttentionKernelGenerator {

    // Typical Llama-3 8B dimensions
    private static final int N_HEADS = 32;
    private static final int HEAD_SIZE = 128;
    private static final int KV_DIM = 1024;  // kvDim = numberOfKVHeads * headSize
    private static final int KV_MUL = 4;     // nHeads / numberOfKVHeads
    private static final int CONTEXT_LENGTH = 8192;
    private static final int LAYER = 0;

    public static void main(String[] args) {
        // Create dummy arrays with realistic sizes
        FloatArray q = new FloatArray(N_HEADS * HEAD_SIZE);
        FloatArray keyCache = new FloatArray(32 * CONTEXT_LENGTH * KV_DIM);  // layers * contextLength * kvDim
        FloatArray valueCache = new FloatArray(32 * CONTEXT_LENGTH * KV_DIM);
        FloatArray xb = new FloatArray(N_HEADS * HEAD_SIZE);
        IntArray positionHolder = new IntArray(1);

        // Initialize position
        positionHolder.set(0, 100);  // Example position

        // Initialize with some dummy data
        for (int i = 0; i < q.getSize(); i++) {
            q.set(i, 0.1f);
        }

        KernelContext context = new KernelContext();

        // Configure grid: nHeads workgroups, headSize local threads
        WorkerGrid1D workerGrid = new WorkerGrid1D(N_HEADS * HEAD_SIZE);
        workerGrid.setLocalWork(HEAD_SIZE, 1, 1);

        // GridScheduler with task name in constructor (TornadoVM 2.2.0 API)
        GridScheduler gridScheduler = new GridScheduler("flashAttention.processHeadsFlashAttention", workerGrid);

        // Create TaskGraph
        TaskGraph taskGraph = new TaskGraph("flashAttention")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, q, keyCache, valueCache, positionHolder)
                .task("processHeadsFlashAttention",
                        TransformerComputeKernelsLayered::processHeadsFlashAttention,
                        context,
                        q, keyCache, valueCache, xb,
                        N_HEADS, HEAD_SIZE, KV_DIM, KV_MUL,
                        positionHolder, LAYER, CONTEXT_LENGTH)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        // Execute (this will trigger kernel compilation and printing with --printKernel)
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler);
            executionPlan.execute();
            System.out.println("Kernel generation complete. Check output for OpenCL kernel.");
        }
    }
}
