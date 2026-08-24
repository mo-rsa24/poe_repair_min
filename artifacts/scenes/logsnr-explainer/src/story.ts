/**
 * The scripted story. Each step is a set of target values and one narration
 * line. Budget: 2 sentences, 40 words. See ~/.claude/SCENE_SIMPLICITY.md.
 */
import type { SceneState } from './store'

export type Step = {
  title: string
  narration: string
  targets: Partial<SceneState>
  lit: string[]
}

export const STORY: Step[] = [
  {
    title: 'Noise is a mixture',
    narration: 'A noisy image is part picture, part fuzz. Drag the slider to change the mix.',
    targets: { alphaBar: 0.5 },
    lit: ['warmup'],
  },
  {
    title: 'log-SNR is the mixing ratio',
    narration:
      'Divide the picture share by the fuzz share, take the log. That is log-SNR, written λ.',
    targets: { alphaBar: 0.92 },
    lit: ['warmup'],
  },
  {
    title: 'The model has 1000 noise levels',
    narration: 'A run uses 50 of them, stepping 20 apart. Each one has a λ.',
    targets: { steps: 50, spacing: 'leading', stepIndex: 0 },
    lit: ['schedule'],
  },
  {
    title: 'Two names for one clock',
    narration: 'λ falls from 7.07 to −5.36, never flat. So step and λ say the same thing.',
    targets: { stepIndex: 25 },
    lit: ['schedule', 'lambda'],
  },
  {
    title: 'Even in steps, uneven in λ',
    narration:
      'Mid-run, one step moves λ by 0.16. The last step moves it by 2.46, fifteen times further.',
    targets: {},
    lit: ['landing'],
  },
  {
    title: 'So F3 uses the step axis',
    narration:
      'On the λ axis that last step is drawn wide and nearly empty. Four of twenty points sit inside it.',
    targets: {},
    lit: ['axes'],
  },
  {
    title: 'What the step axis costs',
    narration:
      'Two samplers reading one curve disagree 0.38 by step, 0.01 by λ. Only matters with a second sampler.',
    targets: { steps: 50, spacing: 'leading', compareSteps: 20, compareSpacing: 'trailing' },
    lit: ['compare'],
  },
]
