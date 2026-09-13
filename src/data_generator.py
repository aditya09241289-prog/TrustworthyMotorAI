import numpy as np
import pandas as pd


FEATURES = ['vibration', 'temperature', 'current', 'rpm']


def generate_dataset(n=800, random_state=42):
    rng = np.random.default_rng(random_state)
    frames = []

    # ---------------------------------------------------------
    # Genuine physical operating states
    # ---------------------------------------------------------

    fault_profiles = {
        'normal': {
            'base': [1.0, 40, 10, 1500],
            'scale': [0.12, 3, 0.8, 35]
        },

        'bearing_fault': {
            'base': [3.4, 58, 12, 1470],
            'scale': [0.45, 5, 1, 45]
        },

        'overload_fault': {
            'base': [2.0, 78, 20, 1400],
            'scale': [0.30, 6, 1.5, 55]
        }
    }

    for fault, profile in fault_profiles.items():

        base = profile['base']
        scale = profile['scale']

        # =====================================================
        # TRUSTED SENSOR DATA
        #
        # Important:
        # Even abnormal values can be trusted if they form a
        # physically consistent motor fault pattern.
        # =====================================================

        trusted = np.column_stack([
            rng.normal(base[i], scale[i], n)
            for i in range(4)
        ])

        trusted_df = pd.DataFrame(
            trusted,
            columns=FEATURES
        )

        trusted_df['fault_label'] = fault
        trusted_df['trust_label'] = 'trusted'

        frames.append(trusted_df)

        # =====================================================
        # TAMPERED SENSOR DATA
        #
        # Start from a genuine physical state and deliberately
        # corrupt one or more sensors independently.
        #
        # This creates inconsistent cyber-physical evidence.
        # =====================================================

        count = n // 2

        tampered = np.column_stack([
            rng.normal(base[i], scale[i], count)
            for i in range(4)
        ])

        for i in range(count):

            tamper_type = rng.choice([
                'single_sensor',
                'multi_sensor',
                'contradictory'
            ])

            # ---------------------------------------------
            # 1. Single sensor spoofing
            # ---------------------------------------------

            if tamper_type == 'single_sensor':

                k = rng.integers(0, 4)

                tamper_amounts = [
                    3.5,   # vibration
                    40.0,  # temperature
                    8.0,   # current
                    400.0  # rpm
                ]

                tampered[i, k] += (
                    rng.choice([-1, 1])
                    * tamper_amounts[k]
                )

            # ---------------------------------------------
            # 2. Multiple independent sensor corruption
            # ---------------------------------------------

            elif tamper_type == 'multi_sensor':

                sensors = rng.choice(
                    4,
                    size=2,
                    replace=False
                )

                tamper_amounts = [
                    3.0,
                    35.0,
                    7.0,
                    350.0
                ]

                for k in sensors:
                    tampered[i, k] += (
                        rng.choice([-1, 1])
                        * tamper_amounts[k]
                    )

            # ---------------------------------------------
            # 3. Physically contradictory evidence
            #
            # Example:
            # extremely high current but artificially normal
            # temperature/vibration, etc.
            # ---------------------------------------------

            else:

                scenario = rng.integers(0, 3)

                if scenario == 0:
                    tampered[i, 2] += 10
                    tampered[i, 1] -= 25

                elif scenario == 1:
                    tampered[i, 0] += 4
                    tampered[i, 1] -= 20

                else:
                    tampered[i, 3] += 450
                    tampered[i, 2] += 8

        tampered_df = pd.DataFrame(
            tampered,
            columns=FEATURES
        )

        tampered_df['fault_label'] = fault
        tampered_df['trust_label'] = 'tampered'

        frames.append(tampered_df)

    # Shuffle the complete dataset
    dataset = pd.concat(
        frames,
        ignore_index=True
    )

    dataset = dataset.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return dataset