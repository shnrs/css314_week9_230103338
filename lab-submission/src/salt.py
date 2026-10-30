import hashlib
import os

def get_student_params(student_id: str):
    """
    Computes salted problem parameters deterministically bound to student_id.
    """
    h = int(hashlib.sha256(student_id.encode('utf-8')).hexdigest()[:8], 16)
    
    # Salted matrix dimension N between 1024 and 1536 (multiple of 64)
    N = 1024 + ((h % 9) * 64)
    
    # Salted stride benchmark buffer size: 2^24 elements (~134 MB double-precision)
    buffer_elements = 16_777_216
    
    # Seed for deterministic random number generation
    seed = h % 100_007
    
    return {
        "student_id": str(student_id),
        "N": N,
        "buffer_elements": buffer_elements,
        "seed": seed
    }

if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "230103338"
    print(get_student_params(sid))