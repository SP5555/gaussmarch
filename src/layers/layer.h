#pragma once

/**
 * Base class for all differentiable layers.
 *
 * Ownership convention:
 *   Forward  : input (non-owning ptr) --> [layer] --> output (owned)
 *   Backward : grad_input (owned)     <-- [layer] <-- grad_output (non-owning ptr)
 *
 * "input" and "output" are always relative to THIS layer, not the direction
 * of flow. This means grad_input is what the layer PRODUCES during backward
 * and sends upstream -- not what it receives.  grad_output is what it
 * RECEIVES from the downstream layer to start the backward computation.
 *
 * In short:
 *   forward()  reads `input`,       writes `output`
 *   backward() reads `grad_output`, writes `grad_input`
 */
class Layer
{
public:
    virtual ~Layer() = default;
    virtual void forward()   = 0;
    virtual void backward()  = 0;
    virtual void zeroGrad() = 0;
};

/**
 * @brief Typed layer base: handles IO wiring and standard lifecycle boilerplate.
 *
 * - TIn      -- forward input type;  non-owning pointer
 * 
 * - TOut     -- forward output type; owned, must have allocate(int n)
 * 
 * - TGradIn  -- grad_input  (dL/d_input),  produced by backward; owned, must have allocate(int n) and zero()
 * 
 * - TGradOut -- grad_output (dL/d_output), received from downstream; non-owning pointer, no requirements
 *
 * Concrete layers inherit this, implement forward()/backward(), and override
 * allocateGrad() only if their grad type needs non-standard allocation arguments
 * (e.g. allocate(n, sh_degree)).
 *
 * Convention for new layers:
 * ```
 *  class MyLayer : public TypedLayer<MyIn, MyOut, MyGradIn, MyGradOut> {
 *      void forward()  override;
 *      void backward() override;
 *      // config setters...
 *  };
 * ```
 */
template<typename TIn, typename TOut, typename TGradIn, typename TGradOut>
class TypedLayer : public Layer
{
public:
    void setInput(const TIn* in_)              { input = in_; }
    TOut&    getOutput()                       { return output; }
    void setGradOutput(const TGradOut* gout_)  { grad_output = gout_; }
    TGradIn& getGradInput()                    { return grad_input; }

    virtual void allocate(int n)     { output.allocate(n); }
    virtual void allocateGrad(int n) { grad_input.allocate(n); }
    void zeroGrad() override        { grad_input.zero(); }

protected:
    const TIn*      input       = nullptr;
    TOut            output;
    const TGradOut* grad_output = nullptr;
    TGradIn         grad_input;
};