def collect_terminus_trajectory(
    cfg,
    env,
    teacher_client,
    task: BenchmarkTask,
    seed: int,
    logger,
    *,
    run_id: str = "",
    teacher_model: str = "",
    teacher_provider: str = "",
) -> "Episode | None":
    """Collect a trajectory using the official Terminus scaffold (JSON batched commands).

    Uses the same prompt template and CommandBatchResponse schema as the
    terminal-bench leaderboard agent, giving leaderboard-comparable results.
    """
    import json as _json
    from pydantic import BaseModel, Field, ConfigDict, ValidationError

    class Command(BaseModel):
        keystrokes: str
        is_blocking: bool
        timeout_sec: float
        model_config = ConfigDict(extra="forbid")

    class CommandBatchResponse(BaseModel):
        state_analysis: str
        explanation: str
        commands: list[Command]
        is_task_complete: bool
        model_config = ConfigDict(extra="forbid")

    TERMINUS_PROMPT = (
        Path(__file__).resolve().parent.parent
        / ".venv312/lib/python3.12/site-packages/terminal_bench/agents/prompt-templates/terminus.txt"
    )
    if not TERMINUS_PROMPT.exists():
        # fallback inline copy
        prompt_template = (
            "You are an AI assistant tasked with solving command-line tasks in a Linux environment. "
            "You will be given a task instruction and the output from previously executed commands. "
            "Your goal is to solve the task by providing batches of shell commands.\n\n"
            "For each response:\n"
            "1. Analyze the current state based on any terminal output provided\n"
            "2. Determine the next set of commands needed to make progress\n"
            "3. Decide if you need to see the output of these commands before proceeding\n\n"
            "Instruction:\n{instruction}\n\n"
            "Your response must be a JSON object that matches this schema:\n\n{response_schema}\n\n"
            "Don't include markdown formatting.\n\n"
            "Note that you operate directly on the terminal from inside a tmux session. "
            "Use tmux keystrokes like `C-x` or `Escape` to interactively navigate the terminal. "
            "If you would like to execute a command that you have written you will need to append "
            "a newline character to the end of your command.\n\n"
            "For example, if you write \"ls -la\" you will need to append a newline character "
            "to the end of your command like this: `ls -la\\n`.\n\n"
            "One thing to be very careful about is handling interactive sessions like less, vim, "
            "or git diff. In these cases, you should not wait for the output of the command. "
            "Instead, you should send the keystrokes to the terminal as if you were typing them.\n\n"
            "The current terminal state is:\n{terminal_state}"
        )
    else:
        prompt_template = TERMINUS_PROMPT.read_text()

    response_schema = _json.dumps(CommandBatchResponse.model_json_schema(), indent=2)

    max_steps = (
        cfg.get("data", {}).get("max_episode_steps")
        or cfg.get("inference", {}).get("step_limit", 50)
    )

    logger.info(f"Collecting trajectory (Terminus): task={task.task_id}, seed={seed}, max_steps={max_steps}")

    try:
        env.reset(task, seed=seed)
    except Exception as e:
        logger.error(f"Failed to reset env for task={task.task_id}: {e}")
        return None

    # Access the underlying tmux session directly
    session = getattr(env, "_session", None)
    if session is None:
        logger.error(f"No tmux session available for task={task.task_id}")
        return None

    steps: list[Step] = []
    total_cost = 0.0
    wall_start = time.time()
    step_idx = 0
    done = False

    # Initial terminal state
    try:
        terminal_state = session.capture_pane(full_history=False) or "[Shell ready]"
    except Exception:
        terminal_state = "[Shell ready]"

    while not done and step_idx < max_steps:
        # Build prompt
        prompt = prompt_template.format(
            instruction=task.goal,
            response_schema=response_schema,
            terminal_state=terminal_state,
            history="",
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            step_start = time.time()
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
            with ThreadPoolExecutor(max_workers=1) as _pool:
                _fut = _pool.submit(teacher_client.chat, messages)
                try:
                    response: LLMResponse = _fut.result(timeout=180)
                except FutureTimeout:
                    logger.error(f"Teacher LLM timed out at step {step_idx}")
                    break
            total_cost += response.cost
            step_dur = time.time() - step_start
            logger.info(
                f"    step {step_idx}: tokens_in={response.input_tokens} "
                f"tokens_out={response.output_tokens} "
                f"cost=${response.cost:.5f} time={step_dur:.1f}s"
            )
        except Exception as e:
            logger.error(f"Teacher LLM call failed at step {step_idx}: {e}")
            break

        # Parse CommandBatchResponse JSON
        raw_text = response.text.strip()
        # Strip markdown code fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        try:
            batch: CommandBatchResponse = CommandBatchResponse.model_validate_json(raw_text)
        except (ValidationError, Exception) as e:
            logger.warning(f"Failed to parse CommandBatchResponse at step {step_idx}: {e}")
            # Record the bad step and continue
            obs_before = terminal_state
            terminal_state = session.capture_pane(full_history=False) or terminal_state
            steps.append(Step(
                step_idx=step_idx,
                observation=Observation(raw_text=obs_before),
                action=Action(raw_text=raw_text, action_type="run"),
                action_source=ActionSource.TEACHER,
                reward=0.0,
                perturbation_type=PerturbationType.NONE,
                llm_tokens_in=response.input_tokens,
                llm_tokens_out=response.output_tokens,
                llm_cost=response.cost,
            ))
            step_idx += 1
            continue

        obs_before = terminal_state

        # Execute each command in the batch via raw tmux keystrokes
        timed_out = False
        for cmd in batch.commands:
            keystrokes = cmd.keystrokes
            # send_keys expects a list or string; handle \n as Enter
            if keystrokes.endswith("\n"):
                keystrokes = keystrokes[:-1]
                try:
                    session.send_keys(
                        [keystrokes, "Enter"],
                        block=cmd.is_blocking,
                        max_timeout_sec=float(cmd.timeout_sec) if cmd.is_blocking else 2.0,
                    )
                except TimeoutError:
                    timed_out = True
                    break
            else:
                # Special keystroke (C-c, Escape, etc.)
                try:
                    session.send_keys(
                        keystrokes,
                        block=False,
                        max_timeout_sec=2.0,
                    )
                except Exception:
                    pass

        # Capture new terminal state
        try:
            terminal_state = session.capture_pane(full_history=False) or ""
        except Exception:
            terminal_state = ""

        # Build action text summarising the batch
        cmds_summary = " && ".join(c.keystrokes.replace("\n", "").strip() for c in batch.commands[:3])
        action_text = f"run [{cmds_summary}]"

        steps.append(Step(
            step_idx=step_idx,
            observation=Observation(raw_text=obs_before),
            action=Action(raw_text=action_text, action_type="run"),
            action_source=ActionSource.TEACHER,
            reward=0.0,
            perturbation_type=PerturbationType.NONE,
            llm_tokens_in=response.input_tokens,
            llm_tokens_out=response.output_tokens,
            llm_cost=response.cost,
        ))
        step_idx += 1

        if batch.is_task_complete or timed_out:
            done = True

    wall_time = time.time() - wall_start

    try:
        score = env.evaluate()
    except Exception as e:
        logger.warning(f"Evaluation failed for task={task.task_id}: {e}")
        score = 0.0
    success = score > 0.5

    metadata = EpisodeMetadata(
        task_id=task.task_id,
        template_id=task.template_id,
        goal=task.goal,
        benchmark="terminalbench",
        site=task.site,
        repo=task.repo,
        difficulty=task.difficulty,
        run_id=run_id,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
    )

    episode = Episode(
        episode_id=f"{task.task_id}_seed{seed}",
        metadata=metadata,
        steps=steps,
        success=success,
        partial_score=score,
        total_cost=total_cost,
        wall_time_seconds=wall_time,
        perturbation_type=PerturbationType.NONE,
    )

    logger.info(
        f"  Task={task.task_id} seed={seed}: "
        f"steps={len(steps)}, success={success}, score={score:.2f}, "
        f"cost=${total_cost:.4f}, time={wall_time:.1f}s"
    )
    return episode


