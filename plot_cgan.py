import re
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# path to your SLURM output file
logfile = r"C:\Users\Aalam Sultanji\Documents\Sem 1b\Pattern Recog\Lab3\Lab3\output_cgan.out"   # change if needed

# regex patterns
epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+of\s+\d+")
gen_pattern = re.compile(r"Generator loss:\s*([0-9.]+)")
disc_pattern = re.compile(r"Discriminator loss:\s*([0-9.]+)")

epochs = []
gen_losses = []
disc_losses = []

with open(logfile, "r") as f:
    for line in f:
        epoch_match = epoch_pattern.search(line)
        gen_match = gen_pattern.search(line)
        disc_match = disc_pattern.search(line)

        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
        if gen_match:
            gen_losses.append(float(gen_match.group(1)))
        if disc_match:
            disc_losses.append(float(disc_match.group(1)))

# sanity check
assert len(epochs) == len(gen_losses) == len(disc_losses), \
    "Mismatch in parsed epochs and losses"

# plot
plt.figure(figsize=(8,6))
plt.plot(epochs, gen_losses, label="Generator loss")
plt.plot(epochs, disc_losses, label="Discriminator loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Losses")
plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.savefig('loss.pdf', dpi=300)
#plt.show()
