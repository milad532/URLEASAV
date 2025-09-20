import wandb

wandb.init(project="Learn-Table-test", job_type="train", config={"seed": 1})

# Create one persistent table
table = wandb.Table(columns=["Input", "Output", "Index"],log_mode = "INCREMENTAL")

# Reset the log mode to allow mutations again
# table._log_mode = "INCREMENTAL"

for i in range(500):
    print(i)
    # Add new rows
    table.add_data(f"in1-{i}", f"out1-{i}", i)
    table.add_data(f"in2-{i}", f"out2-{i}", i)

    # Log the same table object (not a new one!)
    wandb.log({"table": table})

wandb.finish()