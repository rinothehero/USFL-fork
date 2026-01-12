# Model Trainer

Model trainer is a module that handles the training of the model.

> IMPORTANT: the functions in this module must be async and non-blocking. If the functions are not non-blocking, the model training task will not operate asynchronously.

## How to make it as non-blocking?

- Use `asyncio.to_thread` to run the blocking code in a separate thread.

```python
await asyncio.to_thread(optimizer.zero_grad)
await asyncio.to_thread(loss.backward)
await asyncio.to_thread(optimizer.step)
```
