"""Run the teaching dashboard with no arguments: python3 run_training.py."""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
# These are presentation settings, not model hyperparameters.
DISPLAY_DELAY_SECONDS = 0.5
MAX_REFRESHES = 100
MAX_EPOCHS = 300
MATRIX_KEY = 'confusion_rows_actual_columns_predicted_noun_verb'


class Dashboard:
    """Dependency-free terminal renderer; redirected output stays readable."""

    def __init__(self, stream=None, delay=DISPLAY_DELAY_SECONDS):
        self.stream = sys.stdout if stream is None else stream
        self.interactive = self.stream.isatty()
        self.delay = delay if self.interactive else 0.0
        self.lines_drawn = 0
        self.latest = None
        self.last_displayed_epoch = None
        self.refresh_every = max(1, (MAX_EPOCHS + MAX_REFRESHES - 1) // MAX_REFRESHES)

    def draw(self, lines):
        if self.interactive:
            prefix = f'\x1b[{self.lines_drawn}A' if self.lines_drawn else ''
            # Every frame has the same height; no cursor hiding or terminal mode changes.
            frame = prefix + ''.join('\r\x1b[2K' + line + '\n' for line in lines)
            self.stream.write(frame)
            self.lines_drawn = len(lines)
        else:
            self.stream.write('\n'.join(lines) + '\n')
        self.stream.flush()

    @staticmethod
    def matrix_lines(matrix, title):
        return [title, '  Actual \\ Predicted       noun       verb',
                f'  noun                  {matrix[0][0]:7d}    {matrix[0][1]:7d}',
                f'  verb                  {matrix[1][0]:7d}    {matrix[1][1]:7d}']

    def epoch_frame(self, event):
        best = str(event['best_epoch']) if event['best_epoch'] else '-'
        return [f"{event['epoch']:5d}  {event['train_loss']:10.4f} "
                f"{event['train_accuracy']:8.1%}  {event['validation_loss']:10.4f} "
                f"{event['validation_accuracy']:8.1%}  {best:>4}"]

    def __call__(self, event):
        if event['type'] == 'epoch':
            self.latest = event
            epoch = event['epoch']
            interval = self.refresh_every if self.interactive else 10
            if epoch and epoch % interval:
                return
            if self.last_displayed_epoch is None:
                self.draw(['Training',
                           'Epoch  Train loss      Acc    Val loss      Acc  Best'])
                self.lines_drawn = 0
            self.draw(self.epoch_frame(event))
            self.last_displayed_epoch = epoch
            if self.delay:
                time.sleep(self.delay)
        elif event['type'] == 'complete':
            self.complete(event['report'])

    def complete(self, report):
        if self.latest is not None and self.latest['epoch'] != self.last_displayed_epoch:
            self.draw(self.epoch_frame(self.latest))
        # Append a final report; never mix held-out test with live validation.
        self.lines_drawn = 0
        test = report['test']
        self.draw([
            '',
            f"Finished: {report['epochs_run']} epochs | Best: {report['best_epoch']}",
            f"Test loss: {test['loss']:.4f} | Accuracy: {test['accuracy']:.1%}",
            *self.matrix_lines(test[MATRIX_KEY], 'Test confusion matrix'),
            'Saved: results/best_model.npz',
        ])


def main():
    try:
        from train import train
    except ModuleNotFoundError as exc:
        if exc.name == 'numpy':
            raise SystemExit('NumPy is required. See docs/CLI.md for setup.') from None
        raise
    dashboard = Dashboard()
    try:
        train(HERE, HERE / 'results', max_epochs=MAX_EPOCHS,
              callback=dashboard, verbose=False)
    except KeyboardInterrupt:
        # Renderer never hides the cursor or changes terminal modes.
        print('\nTraining interrupted. Any completed best checkpoint remains in results/.')
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
