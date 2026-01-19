"""
SDK-based training for SWE agent using AgentSdkEngine.
This version avoids retokenization by using SQLite trace storage.
"""
import asyncio
import hydra

from rllm.agents.swe_agent import SWEAgent
from rllm.data import DatasetRegistry
from rllm.environments.swe.swe import SWEEnv
from rllm.sdk.shortcuts import get_chat_client
from rllm.trainer.agent_trainer import AgentTrainer


async def run_swe_agent(entry, **kwargs):
    """
    SDK-based SWE agent function that uses traced LLM calls.
    This ensures token IDs are captured from vLLM without retokenization.
    """
    # Initialize environment and agent
    env = SWEEnv(entry=entry)
    agent = SWEAgent(scaffold="r2egym")  # or "sweagent" depending on config
    
    # Get traced chat client that stores token IDs
    client = get_chat_client(base_url="http://localhost:4000/v1", api_key="EMPTY")
    
    try:
        # Reset environment
        observation, info = env.reset()
        agent.reset()
        agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)
        
        max_steps = kwargs.get("max_steps", 50)
        done = False
        total_reward = 0.0
        
        for step_idx in range(max_steps):
            if done:
                break
                
            # Get current messages from agent
            messages = agent.chat_completions
            
            # Make traced LLM call - this captures token IDs in SQLite
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=kwargs.get("model", "Qwen/Qwen3-32B"),
                messages=messages,
                max_tokens=kwargs.get("max_response_length", 8192),
                temperature=kwargs.get("temperature", 1.0),
            )
            
            response_text = response.choices[0].message.content
            
            # Update agent with model response
            action = agent.update_from_model(response_text)
            
            # Take step in environment
            observation, reward, done, info = env.step(action.action)
            total_reward += reward
            
            # Update agent state
            agent.update_from_env(observation=observation, reward=reward, done=done, info=info)
        
        # Close environment
        env.close()
        
        return total_reward
        
    except Exception as e:
        print(f"Error in run_swe_agent: {e}")
        try:
            env.close()
        except:
            pass
        return 0.0


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Load SWE datasets
    train_dataset = DatasetRegistry.load_dataset("R2E_Gym_Subset", "train")
    val_dataset = DatasetRegistry.load_dataset("SWE_Bench_Verified", "test")
    
    assert train_dataset, "Train dataset not found. Please run prepare_swe_data.py first."
    assert val_dataset, "Val dataset not found. Please run prepare_swe_data.py first."
    
    # Create trainer with SDK-based agent function
    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_run_func=run_swe_agent,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
