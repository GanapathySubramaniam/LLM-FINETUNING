from finetuner import finetuner
finetuner_obj=finetuner('meta-llama/Llama-3.2-1B')
prompts = [
    "Explain quantum computing in simple terms.",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis.",
    "What is the significance of the Renaissance?",
    "How does the human immune system work?"
]
responses = [
    "Quantum computing uses quantum bits, or qubits, to perform calculations. Unlike classical bits that are either 0 or 1, qubits can exist in multiple states simultaneously, allowing quantum computers to solve certain complex problems faster.",
    "Renewable energy, such as solar and wind, reduces greenhouse gas emissions, decreases air pollution, and conserves natural resources. It also promotes energy independence and sustainability.",
    "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water. It occurs in the chloroplasts, producing oxygen as a byproduct.",
    "The Renaissance was a cultural movement from the 14th to the 17th century, characterized by a renewed interest in classical art, science, and philosophy. It led to significant advancements in many fields and a shift towards humanism.",
    "The human immune system protects the body from infections and diseases. It consists of physical barriers, immune cells, and proteins that identify and destroy pathogens like bacteria and viruses."
]


finetuner_obj.run(prompts,responses,'./ft1')

prompts = [
    "What is the theory of evolution?",
    "How does blockchain technology work?",
    "Explain the structure of the solar system.",
    "What is the purpose of the United Nations?",
    "How does the process of digestion occur in the human body?"
]
responses = [
    "The theory of evolution, proposed by Charles Darwin, suggests that species change over time through a process called natural selection, where advantageous traits become more common in a population.",
    "Blockchain is a decentralized ledger technology that stores data across a network of computers in a secure, tamper-proof manner. Each 'block' contains data, a timestamp, and is linked to previous blocks, making it nearly impossible to alter past information.",
    "The solar system is made up of the Sun and celestial objects that orbit around it, including eight planets, their moons, and smaller objects like asteroids and comets, all bound by the Sun's gravitational pull.",
    "The United Nations (UN) is an international organization founded in 1945 to promote peace, security, and cooperation among countries. It addresses global issues such as humanitarian aid, conflict resolution, and sustainable development.",
    "Digestion is the process by which the body breaks down food into nutrients. It starts in the mouth, continues in the stomach and intestines, and ends with absorption of nutrients into the bloodstream and elimination of waste."
]
finetuner_obj.run(prompts,responses,'./ft2')
