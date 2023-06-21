import os
from azure.ai.ml import MLClient

def get_credential():
# authentication package
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    
    return credential

def submit_azureml_run():

    ml_client = MLClient(
        subscription_id = "UPDATE WITH YOUR SUBSCRIPTION ID",
        resource_group_name = "UPDATE WITH YOUR RESOURCE GROUP NAME",
        workspace_name = "UPDATE WITH YOUR WORKSPACE NAME",
        credential = get_credential()
    )

    from azure.ai.ml import command
    from azure.ai.ml import Input

    training_job = command(
        # local path where the code is stored
        code="src",
        command = """ds_report \
            && deepspeed main.py \
            --with_aml_log=True \
            --deepspeed ds_config.json
        """,
        inputs={
        },
        environment = "UPDATE WITH YOUR ENV NAME",
        compute = "UPDATE WITH YOUR GPU COMPUTE CLUSTER NAME",
        distribution = {
            "type": "PyTorch",
            # set process count to the number of gpus on the node
            "process_count_per_instance": 1,
        },
        # set instance count to the number of nodes you want to use
        instance_count = 2,
        display_name = "deepspeed",
        description = "prtraining GPT-2 model"
    )

    # submit the job
    returned_job = ml_client.jobs.create_or_update(
        training_job,
        # Project's name
        experiment_name = "gpt-pretrain_v2",
    )

    # get a URL for the status of the job
    print("The url to see your live job running is returned by the sdk:")
    print(returned_job.studio_url)

def main():
    submit_azureml_run()

if __name__ == "__main__":
    main()
